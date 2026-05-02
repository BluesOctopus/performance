from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_diagnostics import apply_stage2_only, build_token_ledger, load_chunks, resolve_encoder_for_name, write_csv
from stage3.alpha_rename import alpha_rename_function_chunk

VARIANTS = ("raw", "stage2_only", "stage2_alpha_rename")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage3-v2 alpha-renaming oracle on function chunks.")
    parser.add_argument("--chunks", required=True, help="Input chunks JSONL from tools/build_py_chunks.py")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--tokenizer", default="gpt4", help="Tokenizer name")
    parser.add_argument(
        "--stage2-profile",
        default="aggressive",
        choices=("safe", "aggressive"),
        help="Stage2 baseline profile",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.chunks)
    resolved = resolve_encoder_for_name(args.tokenizer)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        variant_rows = [
            run_variant(
                variant,
                chunk,
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
                stage2_profile=args.stage2_profile,
            )
            for chunk in chunks
        ]
        detail_rows.extend(variant_rows)
        summary_rows.append(summarize_variant(variant_rows, resolved.tokenizer_name, args.stage2_profile))

    stage2_only_summary = next((row for row in summary_rows if row["variant"] == "stage2_only"), None)
    for row in summary_rows:
        row["delta_vs_stage2_only"] = (
            int(row["effective_saved"]) - int(stage2_only_summary["effective_saved"])
            if stage2_only_summary
            else 0
        )

    write_csv(out_dir / "alpha_rename_oracle_detail.csv", detail_rows)
    write_csv(out_dir / "alpha_rename_oracle_summary.csv", summary_rows)
    return 0


def run_variant(
    variant: str,
    chunk: dict[str, object],
    *,
    tokenizer_name: str,
    encoder,
    tok_type: str,
    stage2_profile: str,
) -> dict[str, object]:
    raw_text = str(chunk["chunk_text"])
    stage2_text = apply_stage2_only(raw_text, stage2_profile=stage2_profile, path=str(chunk["source_id"]))
    renamed_count = 0
    ast_equivalent = variant != "stage2_alpha_rename"
    skipped_reason = ""

    if variant == "raw":
        final_text = raw_text
    elif variant == "stage2_only":
        final_text = stage2_text
    elif variant == "stage2_alpha_rename":
        result = alpha_rename_function_chunk(
            stage2_text,
            tokenizer_name=tokenizer_name,
            encoder=encoder,
            tok_type=tok_type,
        )
        final_text = result.renamed_text
        renamed_count = result.renamed_count
        ast_equivalent = result.ast_equivalent
        skipped_reason = result.skipped_reason
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    ledger = build_token_ledger(
        raw_text,
        final_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=[],
    )
    return {
        "variant": variant,
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "symbol_type": chunk["symbol_type"],
        "symbol_name": chunk["symbol_name"],
        "tokenizer_name": tokenizer_name,
        "stage2_profile": stage2_profile,
        "raw_tokens": ledger.raw_tokens,
        "compressed_tokens": ledger.compressed_tokens,
        "effective_tokens": ledger.effective_tokens,
        "effective_saved": ledger.effective_saved,
        "renamed_count": renamed_count,
        "alpha_rename_hit": renamed_count > 0,
        "ast_equivalent": ast_equivalent,
        "skipped_reason": skipped_reason,
    }


def summarize_variant(rows: list[dict[str, object]], tokenizer_name: str, stage2_profile: str) -> dict[str, object]:
    raw_tokens = sum(int(row["raw_tokens"]) for row in rows)
    compressed_tokens = sum(int(row["compressed_tokens"]) for row in rows)
    effective_tokens = sum(int(row["effective_tokens"]) for row in rows)
    rename_hits = sum(int(bool(row["alpha_rename_hit"])) for row in rows)
    renamed_count_total = sum(int(row["renamed_count"]) for row in rows)
    skipped_counter = Counter(str(row.get("skipped_reason", "") or "") for row in rows)
    return {
        "tokenizer_name": tokenizer_name,
        "stage2_profile": stage2_profile,
        "variant": rows[0]["variant"] if rows else "",
        "raw_tokens": raw_tokens,
        "compressed_tokens": compressed_tokens,
        "effective_tokens": effective_tokens,
        "effective_saved": raw_tokens - effective_tokens,
        "delta_vs_stage2_only": 0,
        "alpha_rename_hit_rate": rename_hits / len(rows) if rows else 0.0,
        "renamed_count_total": renamed_count_total,
        "ast_equivalence_rate": (
            sum(int(bool(row.get("ast_equivalent"))) for row in rows) / len(rows) if rows else 0.0
        ),
        "skipped_reason_distribution": json.dumps(dict(sorted(skipped_counter.items())), ensure_ascii=False),
    }


if __name__ == "__main__":
    raise SystemExit(main())
