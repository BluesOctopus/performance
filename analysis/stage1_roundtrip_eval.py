from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_diagnostics import (
    DEFAULT_STAGE1_MARKER_SCHEME,
    apply_stage1_only,
    build_stage_repo_config,
    build_token_ledger,
    decode_stage1_text,
    load_chunks,
    resolve_encoder_for_name,
    stage1_marker_metrics,
    stage1_vocab_entries_for_text,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage1 deterministic decode and AST equivalence.")
    parser.add_argument("--chunks", required=True, help="Input chunks JSONL from tools/build_py_chunks.py")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--tokenizer", default="gpt4", help="Tokenizer name")
    parser.add_argument(
        "--stage1-marker-scheme",
        default=DEFAULT_STAGE1_MARKER_SCHEME,
        choices=("legacy", "tokenizer_opt"),
        help="Stage1 marker scheme",
    )
    parser.add_argument(
        "--codebook-accounting-mode",
        default="per_chunk",
        choices=("per_chunk", "global_once"),
        help="Codebook accounting mode for output rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.chunks)
    resolved = resolve_encoder_for_name(args.tokenizer)
    repo_config = build_stage_repo_config(
        [chunk["chunk_text"] for chunk in chunks],
        tokenizer_name=resolved.tokenizer_name,
        encoder=resolved.encoder,
        tok_type=resolved.tok_type,
        stage1_marker_scheme=args.stage1_marker_scheme,
        stage3_mode="exact_only",
        enable_b=False,
    )

    rows: list[dict[str, object]] = []
    for chunk in chunks:
        original_text = str(chunk["chunk_text"])
        stage1_text, _stats = apply_stage1_only(
            original_text,
            repo_config,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
        )
        stage1_entries = stage1_vocab_entries_for_text(stage1_text, repo_config)
        if args.codebook_accounting_mode == "global_once":
            stage1_entries = []
        decode_result = decode_stage1_text(stage1_text, repo_config, original_text=original_text)
        ledger = build_token_ledger(
            original_text,
            stage1_text,
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
            codebook_entries=stage1_entries,
        )
        marker_metrics = stage1_marker_metrics(
            stage1_text,
            repo_config,
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
        )
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "symbol_type": chunk["symbol_type"],
                "symbol_name": chunk["symbol_name"],
                "tokenizer_name": resolved.tokenizer_name,
                "tok_type": resolved.tok_type,
                "codebook_accounting_mode": args.codebook_accounting_mode,
                **marker_metrics,
                "roundtrip_ok": decode_result.roundtrip_ok,
                "decode_success": decode_result.decode_success,
                "ast_equivalent": decode_result.ast_equivalent,
                "error_type": decode_result.error_type,
                "raw_tokens": ledger.raw_tokens,
                "stage1_tokens": ledger.compressed_tokens,
                "delta_tokens": ledger.raw_tokens - ledger.compressed_tokens,
                "compressed_tokens": ledger.compressed_tokens,
                "codebook_tokens": ledger.codebook_tokens,
                "wrapper_tokens": ledger.wrapper_tokens,
                "task_prompt_tokens": ledger.task_prompt_tokens,
                "effective_tokens": ledger.effective_tokens,
                "effective_prompt_tokens": ledger.effective_prompt_tokens,
                "gross_saved": ledger.gross_saved,
                "net_saved": ledger.net_saved,
                "effective_saved": ledger.effective_saved,
            }
        )

    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
