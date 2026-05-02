from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_diagnostics import apply_stage2_only, build_token_ledger, load_chunks, resolve_encoder_for_name
from stage3.alpha_pipeline import finalize_alpha_output, run_stage2_alpha_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small Stage2+alpha training manifest without training.")
    parser.add_argument("--chunks", required=True, help="Input chunk jsonl")
    parser.add_argument("--output", required=True, help="Output manifest jsonl")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B", help="Tokenizer")
    parser.add_argument("--stage2-profile", default="aggressive", choices=("safe", "aggressive"))
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of examples")
    parser.add_argument(
        "--filter-mode",
        default="positive_only",
        choices=("all_safe", "positive_only"),
        help="Manifest filtering mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.chunks)
    resolved = resolve_encoder_for_name(args.tokenizer)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        written = 0
        for chunk in chunks:
            if written >= args.limit:
                break
            raw_text = str(chunk["chunk_text"])
            stage2_text = apply_stage2_only(raw_text, stage2_profile=args.stage2_profile, path=str(chunk["source_id"]))
            alpha_result = run_stage2_alpha_pass(
                stage2_text,
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
            )
            finalized = finalize_alpha_output(
                stage2_text,
                alpha_result,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
            )
            if not should_include_record(
                finalized.alpha_metadata,
                finalized.safety_checks,
                filter_mode=args.filter_mode,
            ):
                continue
            ledger = build_token_ledger(
                raw_text,
                finalized.compressed_text,
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
                codebook_entries=[],
            )
            record = {
                "raw_text": raw_text,
                "compressed_text": finalized.compressed_text,
                "variant": "stage2_alpha",
                "tokenizer_name": resolved.tokenizer_name,
                "raw_tokens": ledger.raw_tokens,
                "compressed_tokens": ledger.compressed_tokens,
                "effective_saved": ledger.effective_saved,
                "alpha_metadata": finalized.alpha_metadata,
                "static_vocab_metadata": {},
                "safety_checks": {
                    **finalized.safety_checks,
                    "alpha_ast_ok": bool(finalized.alpha_metadata["alpha_ast_ok"]),
                    "alpha_compile_ok": bool(finalized.alpha_metadata["alpha_compile_ok"]),
                    "alpha_public_signature_preserved": bool(
                        finalized.alpha_metadata["alpha_public_signature_preserved"]
                    ),
                },
                "split": split_for_index(written),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return 0


def split_for_index(index: int) -> str:
    if index % 10 == 0:
        return "test"
    if index % 10 == 1:
        return "dev"
    return "train"


def should_include_record(
    alpha_metadata: dict[str, object],
    safety_checks: dict[str, bool],
    *,
    filter_mode: str,
) -> bool:
    safe = all(
        bool(safety_checks.get(key, False))
        for key in ("compressed_parse_ok", "compressed_compile_ok", "alpha_public_signature_preserved")
    )
    if filter_mode == "all_safe":
        return safe
    return safe and bool(alpha_metadata.get("alpha_applied", False)) and int(alpha_metadata.get("alpha_delta_tokens", 0) or 0) > 0


if __name__ == "__main__":
    raise SystemExit(main())
