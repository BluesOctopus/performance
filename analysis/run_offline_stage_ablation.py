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
    apply_stage3,
    build_stage_repo_config,
    build_token_ledger,
    decode_stage1_text,
    load_chunks,
    resolve_encoder_for_name,
    stage1_vocab_entries_for_text,
    stage3_decode_status,
    summarize_variant,
    with_guardrail,
    write_csv,
)

VARIANTS = (
    "raw",
    "stage2_only",
    "stage1_only",
    "stage3_only",
    "stage1_2",
    "stage2_3",
    "stage1_2_3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Stage1/2/3 ablation on pre-built chunks.")
    parser.add_argument("--chunks", required=True, help="Input chunks JSONL from tools/build_py_chunks.py")
    parser.add_argument("--out_dir", required=True, help="Output directory for detail and summary CSVs")
    parser.add_argument("--tokenizer", default="gpt4", help="Tokenizer name")
    parser.add_argument(
        "--stage2-profile",
        default="aggressive",
        choices=("safe", "aggressive"),
        help="Stage2 baseline profile",
    )
    parser.add_argument(
        "--codebook-accounting-mode",
        default="per_chunk",
        choices=("per_chunk", "global_once"),
        help="Codebook accounting mode for summary and detail rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.chunks)
    resolved = resolve_encoder_for_name(args.tokenizer)
    repo_config = build_stage_repo_config(
        [chunk["chunk_text"] for chunk in chunks],
        tokenizer_name=resolved.tokenizer_name,
        encoder=resolved.encoder,
        tok_type=resolved.tok_type,
        stage3_mode="hybrid",
        enable_b=True,
    )

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        variant_rows = []
        for chunk in chunks:
            row = run_variant(
                variant,
                chunk,
                repo_config,
                resolved.tokenizer_name,
                resolved.encoder,
                resolved.tok_type,
                args.stage2_profile,
                args.codebook_accounting_mode,
            )
            detail_rows.append(row)
            variant_rows.append(row)
        summary_rows.append(
            summarize_variant(
                variant_rows,
                variant=variant,
                tokenizer_name=resolved.tokenizer_name,
                encoder=resolved.encoder,
                tok_type=resolved.tok_type,
                stage2_profile=args.stage2_profile,
                codebook_accounting_mode=args.codebook_accounting_mode,
            )
        )

    write_csv(out_dir / "stage_ablation_offline_detail.csv", detail_rows)
    write_csv(out_dir / "stage_ablation_offline_summary.csv", summary_rows)
    return 0


def run_variant(
    variant: str,
    chunk: dict[str, object],
    repo_config,
    tokenizer_name: str,
    encoder,
    tok_type: str,
    stage2_profile: str,
    codebook_accounting_mode: str,
) -> dict[str, object]:
    original_text = str(chunk["chunk_text"])
    source_id = str(chunk["source_id"])
    stage1_hit = False
    stage3_triggered = False
    roundtrip_ok = False
    ast_equivalent = False
    decode_success = False
    rollback_flag = False
    rollback_reason = ""
    final_text = original_text
    codebook_entries: list[dict[str, object]] = []
    summary_codebook_entries: list[dict[str, object]] = []

    if variant == "raw":
        final_text = original_text

    elif variant == "stage2_only":
        final_text = apply_stage2_only(original_text, stage2_profile=stage2_profile, path=source_id)

    elif variant == "stage1_only":
        final_text, _stats = apply_stage1_only(
            original_text,
            repo_config,
            encoder=encoder,
            tok_type=tok_type,
        )
        stage1_hit = final_text != original_text
        summary_codebook_entries = stage1_vocab_entries_for_text(final_text, repo_config)
        codebook_entries = list(summary_codebook_entries)
        if codebook_accounting_mode == "global_once":
            codebook_entries = []
        decode_result = decode_stage1_text(final_text, repo_config, original_text=original_text)
        roundtrip_ok = decode_result.roundtrip_ok
        ast_equivalent = decode_result.ast_equivalent
        decode_success = decode_result.decode_success

    elif variant == "stage3_only":
        stage3_result = apply_stage3(original_text, repo_config, encoder=encoder, tok_type=tok_type)
        stage3_triggered = bool(stage3_result["stage3_triggered"])
        summary_codebook_entries = list(stage3_result["vocab_entries"])
        codebook_entries = list(summary_codebook_entries)
        if codebook_accounting_mode == "global_once":
            codebook_entries = []
        decode_status = stage3_decode_status(
            repo_config=repo_config,
            encoded_text=stage3_result["text"],
            original_text=original_text,
        )
        roundtrip_ok = decode_status["roundtrip_ok"]
        ast_equivalent = decode_status["ast_equivalent"]
        decode_success = decode_status["decode_success"]
        candidate_ledger, decision = with_guardrail(
            tokenizer_name=tokenizer_name,
            tok_type=tok_type,
            raw_reference_text=original_text,
            baseline_text=original_text,
            baseline_entries=[],
            candidate_text=stage3_result["text"],
            candidate_entries=codebook_entries,
            encoder=encoder,
            rollback_label="rollback_to_raw",
        )
        rollback_flag = decision.should_rollback
        rollback_reason = decision.reason
        final_text = original_text if rollback_flag else stage3_result["text"]
        codebook_entries = [] if rollback_flag else codebook_entries
        return _build_row(
            variant=variant,
            chunk=chunk,
            ledger=candidate_ledger,
            tokenizer_name=tokenizer_name,
            stage2_profile=stage2_profile,
            codebook_accounting_mode=codebook_accounting_mode,
            stage1_hit=stage1_hit,
            stage3_triggered=stage3_triggered,
            roundtrip_ok=roundtrip_ok,
            ast_equivalent=ast_equivalent,
            decode_success=decode_success,
            rollback_flag=rollback_flag,
            rollback_reason=rollback_reason,
            error_type=decode_status["error_type"],
            codebook_entries=summary_codebook_entries if not rollback_flag else [],
        )

    elif variant == "stage1_2":
        adapted = apply_stage1_stage2(
            original_text,
            repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        final_text = adapted["stage2_post_text"]
        stage1_hit = adapted["stage1_text"] != adapted["stage2_pre_text"]
        summary_codebook_entries = stage1_vocab_entries_for_text(adapted["stage1_text"], repo_config)
        codebook_entries = list(summary_codebook_entries)
        if codebook_accounting_mode == "global_once":
            codebook_entries = []
        decode_result = decode_stage1_text(adapted["stage1_text"], repo_config, original_text=adapted["stage2_pre_text"])
        roundtrip_ok = decode_result.roundtrip_ok
        ast_equivalent = decode_result.ast_equivalent
        decode_success = decode_result.decode_success

    elif variant == "stage2_3":
        stage2_text = apply_stage2_only(original_text, stage2_profile=stage2_profile, path=source_id)
        stage3_result = apply_stage3(stage2_text, repo_config, encoder=encoder, tok_type=tok_type)
        stage3_triggered = bool(stage3_result["stage3_triggered"])
        summary_codebook_entries = list(stage3_result["vocab_entries"])
        codebook_entries = list(summary_codebook_entries)
        if codebook_accounting_mode == "global_once":
            codebook_entries = []
        decode_status = stage3_decode_status(
            repo_config=repo_config,
            encoded_text=stage3_result["text"],
            original_text=stage2_text,
        )
        roundtrip_ok = decode_status["roundtrip_ok"]
        ast_equivalent = decode_status["ast_equivalent"]
        decode_success = decode_status["decode_success"]
        ledger, decision = with_guardrail(
            tokenizer_name=tokenizer_name,
            tok_type=tok_type,
            raw_reference_text=original_text,
            baseline_text=stage2_text,
            baseline_entries=[],
            candidate_text=stage3_result["text"],
            candidate_entries=codebook_entries,
            encoder=encoder,
            rollback_label="rollback_to_stage2",
        )
        rollback_flag = decision.should_rollback
        rollback_reason = decision.reason
        return _build_row(
            variant=variant,
            chunk=chunk,
            ledger=ledger,
            tokenizer_name=tokenizer_name,
            stage2_profile=stage2_profile,
            codebook_accounting_mode=codebook_accounting_mode,
            stage1_hit=stage1_hit,
            stage3_triggered=stage3_triggered,
            roundtrip_ok=roundtrip_ok,
            ast_equivalent=ast_equivalent,
            decode_success=decode_success,
            rollback_flag=rollback_flag,
            rollback_reason=rollback_reason,
            error_type=decode_status["error_type"],
            codebook_entries=[] if rollback_flag else summary_codebook_entries,
        )

    elif variant == "stage1_2_3":
        adapted = apply_stage1_stage2(
            original_text,
            repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        stage1_hit = adapted["stage1_text"] != adapted["stage2_pre_text"]
        stage1_entries = stage1_vocab_entries_for_text(adapted["stage1_text"], repo_config)
        summary_stage1_entries = list(stage1_entries)
        summary_codebook_entries = list(stage1_entries)
        decode_result = decode_stage1_text(
            adapted["stage1_text"],
            repo_config,
            original_text=adapted["stage2_pre_text"],
        )
        stage3_result = apply_stage3(adapted["stage2_post_text"], repo_config, encoder=encoder, tok_type=tok_type)
        stage3_triggered = bool(stage3_result["stage3_triggered"])
        stage3_entries = list(stage3_result["vocab_entries"])
        summary_codebook_entries = summary_codebook_entries + stage3_entries
        merged_entries = stage1_entries + stage3_entries
        if codebook_accounting_mode == "global_once":
            stage1_entries = []
            merged_entries = []
        decode_status = stage3_decode_status(
            repo_config=repo_config,
            encoded_text=stage3_result["text"],
            original_text=adapted["stage2_post_text"],
        )
        roundtrip_ok = decode_result.roundtrip_ok and decode_status["roundtrip_ok"]
        ast_equivalent = decode_result.ast_equivalent and decode_status["ast_equivalent"]
        decode_success = decode_result.decode_success and decode_status["decode_success"]
        ledger, decision = with_guardrail(
            tokenizer_name=tokenizer_name,
            tok_type=tok_type,
            raw_reference_text=original_text,
            baseline_text=adapted["stage2_post_text"],
            baseline_entries=stage1_entries,
            candidate_text=stage3_result["text"],
            candidate_entries=merged_entries,
            encoder=encoder,
            rollback_label="rollback_to_stage1_2",
        )
        rollback_flag = decision.should_rollback
        rollback_reason = decision.reason
        return _build_row(
            variant=variant,
            chunk=chunk,
            ledger=ledger,
            tokenizer_name=tokenizer_name,
            stage2_profile=stage2_profile,
            codebook_accounting_mode=codebook_accounting_mode,
            stage1_hit=stage1_hit,
            stage3_triggered=stage3_triggered,
            roundtrip_ok=roundtrip_ok,
            ast_equivalent=ast_equivalent,
            decode_success=decode_success,
            rollback_flag=rollback_flag,
            rollback_reason=rollback_reason,
            error_type=decode_status["error_type"],
            codebook_entries=summary_stage1_entries if rollback_flag else summary_codebook_entries,
        )

    ledger = build_token_ledger(
        original_text,
        final_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=codebook_entries,
    )
    return _build_row(
        variant=variant,
        chunk=chunk,
        ledger=ledger,
        tokenizer_name=tokenizer_name,
        stage2_profile=stage2_profile,
        codebook_accounting_mode=codebook_accounting_mode,
        stage1_hit=stage1_hit,
        stage3_triggered=stage3_triggered,
        roundtrip_ok=roundtrip_ok,
        ast_equivalent=ast_equivalent,
        decode_success=decode_success,
        rollback_flag=rollback_flag,
        rollback_reason=rollback_reason,
        error_type="",
        codebook_entries=summary_codebook_entries or codebook_entries,
    )


def _build_row(
    *,
    variant: str,
    chunk: dict[str, object],
    ledger,
    tokenizer_name: str,
    stage2_profile: str,
    codebook_accounting_mode: str,
    stage1_hit: bool,
    stage3_triggered: bool,
    roundtrip_ok: bool,
    ast_equivalent: bool,
    decode_success: bool,
    rollback_flag: bool,
    rollback_reason: str,
    error_type: str,
    codebook_entries: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "variant": variant,
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "symbol_type": chunk["symbol_type"],
        "symbol_name": chunk["symbol_name"],
        "tokenizer_name": tokenizer_name,
        "stage2_profile": stage2_profile,
        "codebook_accounting_mode": codebook_accounting_mode,
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
        "stage1_hit": stage1_hit,
        "stage3_triggered": stage3_triggered,
        "roundtrip_ok": roundtrip_ok,
        "ast_equivalent": ast_equivalent,
        "decode_success": decode_success,
        "error_type": error_type,
        "rollback_applied": rollback_flag,
        "rollback_reason": rollback_reason,
        "_codebook_entries": codebook_entries,
    }


if __name__ == "__main__":
    raise SystemExit(main())
