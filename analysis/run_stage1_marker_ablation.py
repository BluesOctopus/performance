from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_diagnostics import (
    apply_stage1_only,
    apply_stage1_stage2,
    apply_stage2_only,
    build_stage_repo_config,
    build_token_ledger,
    decode_stage1_text,
    load_chunks,
    resolve_encoder_for_name,
    stage1_marker_metrics,
    stage1_vocab_entries_for_text,
    summarize_variant,
    write_csv,
)

VARIANTS = (
    "raw",
    "stage2_only",
    "stage1_legacy",
    "stage1_tokenizer_opt",
    "stage1_legacy_2",
    "stage1_tokenizer_opt_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy vs tokenizer-aware Stage1 markers.")
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
    chunk_texts = [str(chunk["chunk_text"]) for chunk in chunks]
    repo_configs = {
        "legacy": build_stage_repo_config(
            chunk_texts,
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
            stage1_marker_scheme="legacy",
            stage3_mode="exact_only",
            enable_b=False,
        ),
        "tokenizer_opt": build_stage_repo_config(
            chunk_texts,
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
            stage1_marker_scheme="tokenizer_opt",
            stage3_mode="exact_only",
            enable_b=False,
        ),
    }

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        variant_rows = [
            run_variant(
                variant,
                chunk,
                repo_configs,
                resolved.tokenizer_name,
                resolved.encoder,
                resolved.tok_type,
                args.stage2_profile,
            )
            for chunk in chunks
        ]
        detail_rows.extend(variant_rows)
        summary_rows.append(
            build_summary_row(
                variant_rows,
                variant=variant,
                repo_config=repo_config_for_variant(variant, repo_configs),
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
                stage2_profile=args.stage2_profile,
            )
        )

    stage2_summary = next((row for row in summary_rows if row["variant"] == "stage2_only"), None)
    legacy_summary = next((row for row in summary_rows if row["variant"] == "stage1_legacy"), None)
    tokenizer_opt_summary = next((row for row in summary_rows if row["variant"] == "stage1_tokenizer_opt"), None)
    legacy_2_summary = next((row for row in summary_rows if row["variant"] == "stage1_legacy_2"), None)
    tokenizer_opt_2_summary = next((row for row in summary_rows if row["variant"] == "stage1_tokenizer_opt_2"), None)

    for row in summary_rows:
        row["tokenizer_opt_beats_legacy"] = bool(
            tokenizer_opt_summary and legacy_summary and tokenizer_opt_summary["effective_saved"] > legacy_summary["effective_saved"]
        )
        row["tokenizer_opt_stage1_2_beats_stage2_only"] = bool(
            tokenizer_opt_2_summary and stage2_summary and tokenizer_opt_2_summary["effective_saved"] > stage2_summary["effective_saved"]
        )
        row["tokenizer_opt_stage1_2_delta_vs_stage2_only"] = (
            int(tokenizer_opt_2_summary["effective_saved"]) - int(stage2_summary["effective_saved"])
            if tokenizer_opt_2_summary and stage2_summary
            else 0
        )
        row["tokenizer_opt_delta_vs_legacy"] = (
            int(tokenizer_opt_summary["effective_saved"]) - int(legacy_summary["effective_saved"])
            if tokenizer_opt_summary and legacy_summary
            else 0
        )
        row["tokenizer_opt_2_delta_vs_legacy_2"] = (
            int(tokenizer_opt_2_summary["effective_saved"]) - int(legacy_2_summary["effective_saved"])
            if tokenizer_opt_2_summary and legacy_2_summary
            else 0
        )
        row["stage1_attribution_note"] = attribution_note(row)

    write_csv(out_dir / "stage1_marker_ablation_detail.csv", detail_rows)
    write_csv(out_dir / "stage1_marker_ablation_summary.csv", summary_rows)
    return 0


def run_variant(
    variant: str,
    chunk: dict[str, object],
    repo_configs: dict[str, object],
    tokenizer_name: str,
    encoder,
    tok_type: str,
    stage2_profile: str,
) -> dict[str, object]:
    original_text = str(chunk["chunk_text"])
    source_id = str(chunk["source_id"])
    stage1_hit = False
    roundtrip_ok = False
    ast_equivalent = False
    decode_success = False
    error_type = ""
    codebook_entries: list[dict[str, object]] = []
    stage1_text = ""
    scheme_name = "legacy"
    final_text = original_text

    if variant == "raw":
        final_text = original_text
        repo_config = repo_configs["legacy"]
    elif variant == "stage2_only":
        final_text = apply_stage2_only(original_text, stage2_profile=stage2_profile, path=source_id)
        repo_config = repo_configs["legacy"]
    elif variant == "stage1_legacy":
        repo_config = repo_configs["legacy"]
        stage1_text, _ = apply_stage1_only(original_text, repo_config, encoder=encoder, tok_type=tok_type)
        final_text = stage1_text
        scheme_name = "legacy"
    elif variant == "stage1_tokenizer_opt":
        repo_config = repo_configs["tokenizer_opt"]
        stage1_text, _ = apply_stage1_only(original_text, repo_config, encoder=encoder, tok_type=tok_type)
        final_text = stage1_text
        scheme_name = "tokenizer_opt"
    elif variant == "stage1_legacy_2":
        repo_config = repo_configs["legacy"]
        adapted = apply_stage1_stage2(
            original_text,
            repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        stage1_text = adapted["stage1_text"]
        final_text = adapted["stage2_post_text"]
        scheme_name = "legacy"
    elif variant == "stage1_tokenizer_opt_2":
        repo_config = repo_configs["tokenizer_opt"]
        adapted = apply_stage1_stage2(
            original_text,
            repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        stage1_text = adapted["stage1_text"]
        final_text = adapted["stage2_post_text"]
        scheme_name = "tokenizer_opt"
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    if stage1_text:
        stage1_hit = stage1_text != (apply_stage2_only(original_text, stage2_profile=stage2_profile, path=source_id) if variant.endswith("_2") else original_text)
        decode_result = decode_stage1_text(
            stage1_text,
            repo_config,
            original_text=(apply_stage2_only(original_text, stage2_profile=stage2_profile, path=source_id) if variant.endswith("_2") else original_text),
        )
        roundtrip_ok = decode_result.roundtrip_ok
        ast_equivalent = decode_result.ast_equivalent
        decode_success = decode_result.decode_success
        error_type = decode_result.error_type
        codebook_entries = stage1_vocab_entries_for_text(stage1_text, repo_config)

    ledger = build_token_ledger(
        original_text,
        final_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=codebook_entries,
    )
    marker_metrics = stage1_marker_metrics(
        stage1_text,
        repo_config,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
    )
    repo_diag = stage1_repo_diagnostics(repo_config, [])
    return {
        "variant": variant,
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "symbol_type": chunk["symbol_type"],
        "symbol_name": chunk["symbol_name"],
        "tokenizer_name": tokenizer_name,
        "stage2_profile": stage2_profile,
        "stage1_marker_scheme": scheme_name,
        "raw_tokens": ledger.raw_tokens,
        "compressed_tokens": ledger.compressed_tokens,
        "codebook_tokens": ledger.codebook_tokens,
        "wrapper_tokens": ledger.wrapper_tokens,
        "task_prompt_tokens": ledger.task_prompt_tokens,
        "effective_tokens": ledger.effective_tokens,
        "effective_prompt_tokens": ledger.effective_prompt_tokens,
        "gross_saved": ledger.gross_saved,
        "net_saved": ledger.net_saved,
        "effective_saved": ledger.effective_saved,
        **marker_metrics,
        **repo_diag,
        "stage1_hit": stage1_hit,
        "stage3_triggered": False,
        "roundtrip_ok": roundtrip_ok,
        "ast_equivalent": ast_equivalent,
        "decode_success": decode_success,
        "error_type": error_type,
        "rollback_applied": False,
        "rollback_reason": "",
        "_codebook_entries": codebook_entries,
    }


def repo_config_for_variant(variant: str, repo_configs: dict[str, object]):
    if "tokenizer_opt" in variant:
        return repo_configs["tokenizer_opt"]
    return repo_configs["legacy"]


def build_summary_row(
    rows: list[dict[str, object]],
    *,
    variant: str,
    repo_config,
    tokenizer_name: str,
    encoder,
    tok_type: str,
    stage2_profile: str,
) -> dict[str, object]:
    summary = summarize_variant(
        rows,
        variant=variant,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        stage2_profile=stage2_profile,
        codebook_accounting_mode="per_chunk",
    )
    summary.update(stage1_repo_diagnostics(repo_config, rows))
    summary["stage1_attribution_note"] = attribution_note(summary)
    return summary


def stage1_repo_diagnostics(repo_config, rows: list[dict[str, object]]) -> dict[str, object]:
    candidate_count = len(getattr(repo_config, "stage1_candidate_stats", []) or [])
    selected_count = len(getattr(repo_config, "stage1_selected_stats", []) or [])
    apply_hit_count = sum(int(bool(row.get("stage1_hit"))) for row in rows)
    apply_miss_count = len(rows) - apply_hit_count if rows and any("stage1_" in str(row.get("variant", "")) for row in rows) else 0
    return {
        "stage1_candidate_count": candidate_count,
        "stage1_selected_count": selected_count,
        "stage1_selected_skeleton_count": len(getattr(repo_config, "stage1_marker_tokens", []) or []),
        "stage1_rejected_no_gain_count": int(getattr(repo_config, "stage1_rejected_no_gain_count", 0) or 0),
        "stage1_rejected_intro_cost_count": int(getattr(repo_config, "stage1_rejected_intro_cost_count", 0) or 0),
        "stage1_rejected_marker_cost_count": int(getattr(repo_config, "stage1_rejected_marker_cost_count", 0) or 0),
        "stage1_rejected_low_frequency_count": int(getattr(repo_config, "stage1_low_frequency_count", 0) or 0),
        "stage1_apply_hit_count": apply_hit_count,
        "stage1_apply_miss_count": apply_miss_count,
    }


def attribution_note(summary: dict[str, object]) -> str:
    hit_rate = float(summary.get("stage1_hit_rate", 0.0) or 0.0)
    if hit_rate > 0.0:
        return ""
    candidate_count = int(summary.get("stage1_candidate_count", 0) or 0)
    selected_count = int(summary.get("stage1_selected_count", 0) or 0)
    apply_hit_count = int(summary.get("stage1_apply_hit_count", 0) or 0)
    marker_rejects = int(summary.get("stage1_rejected_marker_cost_count", 0) or 0)
    intro_rejects = int(summary.get("stage1_rejected_intro_cost_count", 0) or 0)
    if candidate_count == 0:
        return "stage1_hit_rate=0; no Stage1 candidates"
    if selected_count == 0:
        return "stage1_hit_rate=0; candidates exist but none selected"
    if apply_hit_count == 0:
        return "stage1_hit_rate=0; selected skeletons were never applied"
    if marker_rejects > 0:
        return "stage1_hit_rate=0; marker cost blocked net gains"
    if intro_rejects > 0:
        return "stage1_hit_rate=0; codebook intro cost absorbed gains"
    return "stage1_hit_rate=0; cannot attribute gains to Stage1"


if __name__ == "__main__":
    raise SystemExit(main())
