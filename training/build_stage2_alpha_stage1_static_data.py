from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_diagnostics import (
    apply_stage1_only,
    apply_stage2_only,
    build_stage_repo_config,
    build_token_ledger,
    decode_stage1_text,
    load_chunks,
    resolve_encoder_for_name,
)
from stage1_static.static_vocab import build_static_vocab_manifest, restrict_repo_config_to_static_manifest
from stage3.alpha_pipeline import finalize_alpha_output, run_stage2_alpha_pass, safe_compile_ok, safe_parse_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small Stage2+alpha+Stage1-static manifest without training.")
    parser.add_argument("--chunks", required=True, help="Input chunk jsonl")
    parser.add_argument("--output", required=True, help="Output manifest jsonl")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B", help="Tokenizer")
    parser.add_argument("--stage2-profile", default="aggressive", choices=("safe", "aggressive"))
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of examples")
    parser.add_argument("--top-k", type=int, default=32, help="Static vocab top-k")
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
    static_manifest = build_static_vocab_manifest(
        chunks,
        tokenizer_name=resolved.tokenizer_name,
        encoder=resolved.encoder,
        tok_type=resolved.tok_type,
        top_k=args.top_k,
    )
    repo_config = build_stage_repo_config(
        [str(chunk["chunk_text"]) for chunk in chunks],
        tokenizer_name=resolved.tokenizer_name,
        encoder=resolved.encoder,
        tok_type=resolved.tok_type,
        stage1_marker_scheme="tokenizer_opt",
        stage3_mode="exact_only",
        enable_b=False,
    )
    static_repo_config = restrict_repo_config_to_static_manifest(repo_config, static_manifest)
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
            finalized_alpha = finalize_alpha_output(
                stage2_text,
                alpha_result,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
            )
            static_text, _ = apply_stage1_only(
                finalized_alpha.compressed_text,
                static_repo_config,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
            )
            decode_result = decode_stage1_text(
                static_text,
                static_repo_config,
                original_text=finalized_alpha.compressed_text,
            )
            encoded_parse_ok = safe_parse_ok(static_text)
            decoded_parse_ok = safe_parse_ok(decode_result.decoded_text) if decode_result.decode_success else False
            decoded_compile_ok = safe_compile_ok(decode_result.decoded_text) if decode_result.decode_success else False
            static_metadata = {
                "top_k": static_manifest["top_k"],
                "entry_count": static_manifest["entry_count"],
                "encoded_parse_ok": encoded_parse_ok,
                "decode_success": bool(decode_result.decode_success),
                "decoded_parse_ok": decoded_parse_ok,
                "decoded_compile_ok": decoded_compile_ok,
                "decoded_ast_equivalent": bool(decode_result.ast_equivalent),
                "decoded_roundtrip_ok": bool(decode_result.roundtrip_ok),
                "decoded_error_type": str(decode_result.error_type),
                "decoded_text": decode_result.decoded_text,
            }
            if not should_include_record(
                finalized_alpha.alpha_metadata,
                finalized_alpha.safety_checks,
                static_metadata,
                filter_mode=args.filter_mode,
            ):
                continue
            ledger = build_token_ledger(
                raw_text,
                static_text,
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
                codebook_entries=[],
            )
            record = {
                "raw_text": raw_text,
                "compressed_text": static_text,
                "variant": "stage2_alpha_stage1_static",
                "tokenizer_name": resolved.tokenizer_name,
                "raw_tokens": ledger.raw_tokens,
                "compressed_tokens": ledger.compressed_tokens,
                "effective_saved": ledger.effective_saved,
                "alpha_metadata": finalized_alpha.alpha_metadata,
                "static_vocab_metadata": static_metadata,
                "safety_checks": {
                    **finalized_alpha.safety_checks,
                    "alpha_ast_ok": bool(finalized_alpha.alpha_metadata["alpha_ast_ok"]),
                    "alpha_compile_ok": bool(finalized_alpha.alpha_metadata["alpha_compile_ok"]),
                    "alpha_public_signature_preserved": bool(
                        finalized_alpha.alpha_metadata["alpha_public_signature_preserved"]
                    ),
                    "encoded_parse_ok": encoded_parse_ok,
                    "static_vocab_decode_success": bool(decode_result.decode_success),
                    "decoded_parse_ok": decoded_parse_ok,
                    "decoded_compile_ok": decoded_compile_ok,
                    "decoded_ast_equivalent": bool(decode_result.ast_equivalent),
                },
                "split": split_for_index(written),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    manifest_path = output_path.with_suffix(output_path.suffix + ".static_vocab_manifest.json")
    manifest_path.write_text(json.dumps(static_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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
    static_metadata: dict[str, object],
    *,
    filter_mode: str,
) -> bool:
    safe = all(
        bool(safety_checks.get(key, False))
        for key in ("compressed_parse_ok", "compressed_compile_ok", "alpha_public_signature_preserved")
    )
    decoded_safe = bool(static_metadata.get("decode_success", False)) and bool(
        static_metadata.get("decoded_parse_ok", False)
    ) and bool(static_metadata.get("decoded_compile_ok", False)) and bool(
        static_metadata.get("decoded_ast_equivalent", False)
    )
    if filter_mode == "all_safe":
        return safe and decoded_safe
    return (
        safe
        and decoded_safe
        and bool(alpha_metadata.get("alpha_applied", False))
        and int(alpha_metadata.get("alpha_delta_tokens", 0) or 0) > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
