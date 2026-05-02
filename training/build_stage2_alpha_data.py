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
from stage3.alpha_pipeline import run_stage2_alpha_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small Stage2+alpha training manifest without training.")
    parser.add_argument("--chunks", required=True, help="Input chunk jsonl")
    parser.add_argument("--output", required=True, help="Output manifest jsonl")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B", help="Tokenizer")
    parser.add_argument("--stage2-profile", default="aggressive", choices=("safe", "aggressive"))
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of examples")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.chunks)[: args.limit]
    resolved = resolve_encoder_for_name(args.tokenizer)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, chunk in enumerate(chunks):
            raw_text = str(chunk["chunk_text"])
            stage2_text = apply_stage2_only(raw_text, stage2_profile=args.stage2_profile, path=str(chunk["source_id"]))
            alpha_result = run_stage2_alpha_pass(
                stage2_text,
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
            )
            ledger = build_token_ledger(
                raw_text,
                alpha_result.output_text,
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
                codebook_entries=[],
            )
            record = {
                "raw_text": raw_text,
                "compressed_text": alpha_result.output_text,
                "variant": "stage2_alpha",
                "tokenizer_name": resolved.tokenizer_name,
                "raw_tokens": ledger.raw_tokens,
                "compressed_tokens": ledger.compressed_tokens,
                "effective_saved": ledger.effective_saved,
                "alpha_metadata": alpha_result.metadata.to_dict(),
                "static_vocab_metadata": {},
                "safety_checks": {
                    "alpha_ast_ok": alpha_result.metadata.alpha_ast_ok,
                    "alpha_compile_ok": alpha_result.metadata.alpha_compile_ok,
                    "alpha_public_signature_preserved": alpha_result.metadata.alpha_public_signature_preserved,
                },
                "split": split_for_index(index),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


def split_for_index(index: int) -> str:
    if index % 10 == 0:
        return "test"
    if index % 10 == 1:
        return "dev"
    return "train"


if __name__ == "__main__":
    raise SystemExit(main())
