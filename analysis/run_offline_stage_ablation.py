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
    apply_stage1_stage2,
    apply_stage2_only,
    apply_stage3,
    build_stage_repo_config,
    build_token_ledger,
    decode_stage1_text,
    load_chunks,
    resolve_encoder_for_name,
    stage1_marker_metrics,
    stage1_vocab_entries_for_text,
    stage3_decode_status,
    summarize_variant,
    with_guardrail,
    write_csv,
)
from stage3.alpha_pipeline import run_stage2_alpha_pass

VARIANTS = (
    "raw",
    "stage2_only",
    "stage1_only",
    "stage3_only",
    "stage1_2",
    "stage2_3",
    "stage1_2_3",
    "stage2_alpha",
    "stage2_alpha_stage1_tokenizer_opt",
    "stage2_alpha_stage1_static",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Stage1/2/3 ablation on pre-built chunks.")
    parser.add_argument("--chunks", required=True, help="Input chunks JSONL from tools/build_py_chunks.py")
    parser.add_argument("--out_dir", required=True, help="Output directory for detail and summary CSVs")
    parser.add_argument("--tokenizer", default="gpt4", help="Tokenizer name")
    parser.add_argument(
        "--stage1-marker-scheme",
        default=DEFAULT_STAGE1_MARKER_SCHEME,
        choices=("legacy", "tokenizer_opt"),
        help="Stage1 marker scheme",
    )
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
    chunk_texts = [str(chunk["chunk_text"]) for chunk in chunks]
    repo_configs = {
        "selected": build_stage_repo_config(
            chunk_texts,
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
            stage1_marker_scheme=args.stage1_marker_scheme,
            stage3_mode="hybrid",
            enable_b=True,
        ),
        "tokenizer_opt": build_stage_repo_config(
            chunk_texts,
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
            stage1_marker_scheme="tokenizer_opt",
            stage3_mode="hybrid",
            enable_b=True,
        ),
    }

    detail_rows: list[dict[str, object]] = []
    variant_rows_map: dict[str, list[dict[str, object]]] = {}
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
                args.codebook_accounting_mode,
                args.stage1_marker_scheme,
            )
            for chunk in chunks
        ]
        variant_rows_map[variant] = variant_rows
        detail_rows.extend(variant_rows)

    stage2_only_rows = variant_rows_map["stage2_only"]
    stage2_baseline_by_chunk = {
        str(row["chunk_id"]): int(row["effective_tokens"])
        for row in stage2_only_rows
    }
    summary_rows = [
        build_summary_row(
            variant_rows_map[variant],
            variant=variant,
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
            stage2_profile=args.stage2_profile,
            codebook_accounting_mode=args.codebook_accounting_mode,
            stage2_baseline_by_chunk=stage2_baseline_by_chunk,
        )
        for variant in VARIANTS
    ]

    write_csv(out_dir / "stage_ablation_offline_detail.csv", detail_rows)
    write_csv(out_dir / "stage_ablation_offline_summary.csv", summary_rows)
    return 0


def run_variant(
    variant: str,
    chunk: dict[str, object],
    repo_configs: dict[str, object],
    tokenizer_name: str,
    encoder,
    tok_type: str,
    stage2_profile: str,
    codebook_accounting_mode: str,
    stage1_marker_scheme: str,
) -> dict[str, object]:
    original_text = str(chunk["chunk_text"])
    source_id = str(chunk["source_id"])
    selected_repo_config = repo_configs["selected"]
    token_opt_repo_config = repo_configs["tokenizer_opt"]
    stage2_text = apply_stage2_only(original_text, stage2_profile=stage2_profile, path=source_id)

    stage1_hit = False
    stage3_triggered = False
    roundtrip_ok = False
    ast_equivalent = False
    decode_success = False
    rollback_flag = False
    rollback_reason = ""
    stage1_marker_text = ""
    codebook_entries: list[dict[str, object]] = []
    summary_codebook_entries: list[dict[str, object]] = []
    alpha_eligible_function = False
    alpha_hit = False
    alpha_renamed_count = 0
    alpha_metadata: dict[str, object] = default_alpha_metadata()
    final_text = original_text
    error_type = ""

    if variant == "raw":
        final_text = original_text

    elif variant == "stage2_only":
        final_text = stage2_text

    elif variant == "stage1_only":
        final_text, _stats = apply_stage1_only(
            original_text,
            selected_repo_config,
            encoder=encoder,
            tok_type=tok_type,
        )
        stage1_marker_text = final_text
        stage1_hit = final_text != original_text
        summary_codebook_entries = stage1_vocab_entries_for_text(final_text, selected_repo_config)
        codebook_entries = [] if codebook_accounting_mode == "global_once" else list(summary_codebook_entries)
        decode_result = decode_stage1_text(final_text, selected_repo_config, original_text=original_text)
        roundtrip_ok = decode_result.roundtrip_ok
        ast_equivalent = decode_result.ast_equivalent
        decode_success = decode_result.decode_success
        error_type = decode_result.error_type

    elif variant == "stage3_only":
        stage3_result = apply_stage3(original_text, selected_repo_config, encoder=encoder, tok_type=tok_type)
        stage3_triggered = bool(stage3_result["stage3_triggered"])
        summary_codebook_entries = list(stage3_result["vocab_entries"])
        codebook_entries = [] if codebook_accounting_mode == "global_once" else list(summary_codebook_entries)
        decode_status = stage3_decode_status(
            repo_config=selected_repo_config,
            encoded_text=stage3_result["text"],
            original_text=original_text,
        )
        roundtrip_ok = decode_status["roundtrip_ok"]
        ast_equivalent = decode_status["ast_equivalent"]
        decode_success = decode_status["decode_success"]
        error_type = decode_status["error_type"]
        ledger, decision = with_guardrail(
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
        return build_row(
            variant,
            chunk,
            ledger,
            tokenizer_name,
            stage2_profile,
            codebook_accounting_mode,
            stage1_marker_scheme,
            stage1_marker_metrics("", selected_repo_config, tokenizer_name=tokenizer_name, encoder=encoder, tok_type=tok_type),
            stage1_hit,
            stage3_triggered,
            roundtrip_ok,
            ast_equivalent,
            decode_success,
            rollback_flag,
            rollback_reason,
            error_type,
            [] if rollback_flag else summary_codebook_entries,
            alpha_eligible_function,
            alpha_hit,
            alpha_renamed_count,
        )

    elif variant == "stage1_2":
        adapted = apply_stage1_stage2(
            original_text,
            selected_repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        final_text = adapted["stage2_post_text"]
        stage1_marker_text = adapted["stage1_text"]
        stage1_hit = adapted["stage1_text"] != adapted["stage2_pre_text"]
        summary_codebook_entries = stage1_vocab_entries_for_text(adapted["stage1_text"], selected_repo_config)
        codebook_entries = [] if codebook_accounting_mode == "global_once" else list(summary_codebook_entries)
        decode_result = decode_stage1_text(adapted["stage1_text"], selected_repo_config, original_text=adapted["stage2_pre_text"])
        roundtrip_ok = decode_result.roundtrip_ok
        ast_equivalent = decode_result.ast_equivalent
        decode_success = decode_result.decode_success
        error_type = decode_result.error_type

    elif variant == "stage2_3":
        stage3_result = apply_stage3(stage2_text, selected_repo_config, encoder=encoder, tok_type=tok_type)
        stage3_triggered = bool(stage3_result["stage3_triggered"])
        summary_codebook_entries = list(stage3_result["vocab_entries"])
        codebook_entries = [] if codebook_accounting_mode == "global_once" else list(summary_codebook_entries)
        decode_status = stage3_decode_status(
            repo_config=selected_repo_config,
            encoded_text=stage3_result["text"],
            original_text=stage2_text,
        )
        roundtrip_ok = decode_status["roundtrip_ok"]
        ast_equivalent = decode_status["ast_equivalent"]
        decode_success = decode_status["decode_success"]
        error_type = decode_status["error_type"]
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
        return build_row(
            variant,
            chunk,
            ledger,
            tokenizer_name,
            stage2_profile,
            codebook_accounting_mode,
            stage1_marker_scheme,
            stage1_marker_metrics("", selected_repo_config, tokenizer_name=tokenizer_name, encoder=encoder, tok_type=tok_type),
            stage1_hit,
            stage3_triggered,
            roundtrip_ok,
            ast_equivalent,
            decode_success,
            rollback_flag,
            rollback_reason,
            error_type,
            [] if rollback_flag else summary_codebook_entries,
            alpha_eligible_function,
            alpha_hit,
            alpha_renamed_count,
        )

    elif variant == "stage1_2_3":
        adapted = apply_stage1_stage2(
            original_text,
            selected_repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        stage1_marker_text = adapted["stage1_text"]
        stage1_hit = adapted["stage1_text"] != adapted["stage2_pre_text"]
        stage1_entries = stage1_vocab_entries_for_text(adapted["stage1_text"], selected_repo_config)
        decode_result = decode_stage1_text(adapted["stage1_text"], selected_repo_config, original_text=adapted["stage2_pre_text"])
        stage3_result = apply_stage3(adapted["stage2_post_text"], selected_repo_config, encoder=encoder, tok_type=tok_type)
        stage3_triggered = bool(stage3_result["stage3_triggered"])
        stage3_entries = list(stage3_result["vocab_entries"])
        summary_codebook_entries = list(stage1_entries) + stage3_entries
        merged_entries = [] if codebook_accounting_mode == "global_once" else list(stage1_entries) + stage3_entries
        baseline_entries = [] if codebook_accounting_mode == "global_once" else list(stage1_entries)
        decode_status = stage3_decode_status(
            repo_config=selected_repo_config,
            encoded_text=stage3_result["text"],
            original_text=adapted["stage2_post_text"],
        )
        roundtrip_ok = decode_result.roundtrip_ok and decode_status["roundtrip_ok"]
        ast_equivalent = decode_result.ast_equivalent and decode_status["ast_equivalent"]
        decode_success = decode_result.decode_success and decode_status["decode_success"]
        error_type = decode_status["error_type"]
        ledger, decision = with_guardrail(
            tokenizer_name=tokenizer_name,
            tok_type=tok_type,
            raw_reference_text=original_text,
            baseline_text=adapted["stage2_post_text"],
            baseline_entries=baseline_entries,
            candidate_text=stage3_result["text"],
            candidate_entries=merged_entries,
            encoder=encoder,
            rollback_label="rollback_to_stage1_2",
        )
        rollback_flag = decision.should_rollback
        rollback_reason = decision.reason
        return build_row(
            variant,
            chunk,
            ledger,
            tokenizer_name,
            stage2_profile,
            codebook_accounting_mode,
            stage1_marker_scheme,
            stage1_marker_metrics(stage1_marker_text, selected_repo_config, tokenizer_name=tokenizer_name, encoder=encoder, tok_type=tok_type),
            stage1_hit,
            stage3_triggered,
            roundtrip_ok,
            ast_equivalent,
            decode_success,
            rollback_flag,
            rollback_reason,
            error_type,
            stage1_entries if rollback_flag else summary_codebook_entries,
            alpha_eligible_function,
            alpha_hit,
            alpha_renamed_count,
        )

    elif variant == "stage2_alpha":
        alpha_result = run_stage2_alpha_pass(
            stage2_text,
            tokenizer_name=tokenizer_name,
            encoder=encoder,
            tok_type=tok_type,
        )
        alpha_metadata = alpha_result.metadata.to_dict()
        alpha_eligible_function = alpha_metadata["alpha_skipped_reason"] in {"", "no_eligible_locals"}
        alpha_hit = bool(alpha_metadata["alpha_applied"])
        alpha_renamed_count = int(alpha_metadata["alpha_renamed_count"])
        final_text = alpha_result.output_text
        ast_equivalent = bool(alpha_metadata["alpha_ast_ok"] and alpha_metadata["alpha_public_signature_preserved"])
        decode_success = bool(alpha_metadata["alpha_compile_ok"])
        roundtrip_ok = ast_equivalent
        error_type = str(alpha_metadata["alpha_skipped_reason"] or alpha_metadata["alpha_rollback_reason"])

    elif variant == "stage2_alpha_stage1_tokenizer_opt":
        alpha_result = run_stage2_alpha_pass(
            stage2_text,
            tokenizer_name=tokenizer_name,
            encoder=encoder,
            tok_type=tok_type,
        )
        alpha_metadata = alpha_result.metadata.to_dict()
        alpha_eligible_function = alpha_metadata["alpha_skipped_reason"] in {"", "no_eligible_locals"}
        alpha_hit = bool(alpha_metadata["alpha_applied"])
        alpha_renamed_count = int(alpha_metadata["alpha_renamed_count"])
        stage1_marker_text, _stats = apply_stage1_only(
            alpha_result.output_text,
            token_opt_repo_config,
            encoder=encoder,
            tok_type=tok_type,
        )
        final_text = stage1_marker_text
        stage1_hit = stage1_marker_text != alpha_result.output_text
        summary_codebook_entries = stage1_vocab_entries_for_text(stage1_marker_text, token_opt_repo_config)
        codebook_entries = [] if codebook_accounting_mode == "global_once" else list(summary_codebook_entries)
        decode_result = decode_stage1_text(stage1_marker_text, token_opt_repo_config, original_text=alpha_result.output_text)
        roundtrip_ok = bool(alpha_metadata["alpha_ast_ok"]) and decode_result.roundtrip_ok
        ast_equivalent = bool(alpha_metadata["alpha_ast_ok"]) and decode_result.ast_equivalent
        decode_success = bool(alpha_metadata["alpha_compile_ok"]) and decode_result.decode_success
        error_type = str(alpha_metadata["alpha_skipped_reason"] or alpha_metadata["alpha_rollback_reason"] or decode_result.error_type)

    elif variant == "stage2_alpha_stage1_static":
        alpha_result = run_stage2_alpha_pass(
            stage2_text,
            tokenizer_name=tokenizer_name,
            encoder=encoder,
            tok_type=tok_type,
        )
        alpha_metadata = alpha_result.metadata.to_dict()
        alpha_eligible_function = alpha_metadata["alpha_skipped_reason"] in {"", "no_eligible_locals"}
        alpha_hit = bool(alpha_metadata["alpha_applied"])
        alpha_renamed_count = int(alpha_metadata["alpha_renamed_count"])
        stage1_marker_text, _stats = apply_stage1_only(
            alpha_result.output_text,
            token_opt_repo_config,
            encoder=encoder,
            tok_type=tok_type,
        )
        final_text = stage1_marker_text
        stage1_hit = stage1_marker_text != alpha_result.output_text
        summary_codebook_entries = []
        codebook_entries = []
        decode_result = decode_stage1_text(stage1_marker_text, token_opt_repo_config, original_text=alpha_result.output_text)
        roundtrip_ok = bool(alpha_metadata["alpha_ast_ok"]) and decode_result.roundtrip_ok
        ast_equivalent = bool(alpha_metadata["alpha_ast_ok"]) and decode_result.ast_equivalent
        decode_success = bool(alpha_metadata["alpha_compile_ok"]) and decode_result.decode_success
        error_type = str(alpha_metadata["alpha_skipped_reason"] or alpha_metadata["alpha_rollback_reason"] or decode_result.error_type)

    ledger = build_token_ledger(
        original_text,
        final_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=codebook_entries,
    )
    repo_config_for_metrics = token_opt_repo_config if "tokenizer_opt" in variant or variant.endswith("_static") else selected_repo_config
    return build_row(
        variant,
        chunk,
        ledger,
        tokenizer_name,
        stage2_profile,
        codebook_accounting_mode,
        stage1_marker_scheme,
        stage1_marker_metrics(stage1_marker_text, repo_config_for_metrics, tokenizer_name=tokenizer_name, encoder=encoder, tok_type=tok_type),
        stage1_hit,
        stage3_triggered,
        roundtrip_ok,
        ast_equivalent,
        decode_success,
        rollback_flag,
        rollback_reason,
        error_type,
        summary_codebook_entries or codebook_entries,
        alpha_eligible_function,
        alpha_hit,
        alpha_renamed_count,
        alpha_metadata,
    )


def build_row(
    variant: str,
    chunk: dict[str, object],
    ledger,
    tokenizer_name: str,
    stage2_profile: str,
    codebook_accounting_mode: str,
    stage1_marker_scheme: str,
    marker_metrics: dict[str, object],
    stage1_hit: bool,
    stage3_triggered: bool,
    roundtrip_ok: bool,
    ast_equivalent: bool,
    decode_success: bool,
    rollback_flag: bool,
    rollback_reason: str,
    error_type: str,
    codebook_entries: list[dict[str, object]],
    alpha_eligible_function: bool,
    alpha_hit: bool,
    alpha_renamed_count: int,
    alpha_metadata: dict[str, object],
) -> dict[str, object]:
    variant_status, train_allowed, train_block_reason = variant_training_policy(variant)
    return {
        "variant": variant,
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "symbol_type": chunk["symbol_type"],
        "symbol_name": chunk["symbol_name"],
        "tokenizer_name": tokenizer_name,
        "stage2_profile": stage2_profile,
        "codebook_accounting_mode": codebook_accounting_mode,
        "stage1_marker_scheme": stage1_marker_scheme,
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
        "stage1_hit": stage1_hit,
        "stage3_triggered": stage3_triggered,
        "roundtrip_ok": roundtrip_ok,
        "ast_equivalent": ast_equivalent,
        "decode_success": decode_success,
        "error_type": error_type,
        "rollback_applied": rollback_flag,
        "rollback_reason": rollback_reason,
        "variant_status": variant_status,
        "train_allowed": train_allowed,
        "train_block_reason": train_block_reason,
        "alpha_eligible_function": alpha_eligible_function,
        "alpha_hit": alpha_hit,
        "alpha_renamed_count": alpha_renamed_count,
        **alpha_metadata,
        "_codebook_entries": codebook_entries,
    }


def build_summary_row(
    rows: list[dict[str, object]],
    *,
    variant: str,
    tokenizer_name: str,
    encoder,
    tok_type: str,
    stage2_profile: str,
    codebook_accounting_mode: str,
    stage2_baseline_by_chunk: dict[str, int],
) -> dict[str, object]:
    summary = summarize_variant(
        rows,
        variant=variant,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        stage2_profile=stage2_profile,
        codebook_accounting_mode=codebook_accounting_mode,
    )
    deltas: list[int] = []
    eligible_deltas: list[int] = []
    eligible_rows = [row for row in rows if bool(row.get("alpha_eligible_function"))]
    for row in rows:
        baseline = stage2_baseline_by_chunk.get(str(row["chunk_id"]), int(row["effective_tokens"]))
        delta = baseline - int(row["effective_tokens"])
        row["paired_delta_vs_stage2_only"] = delta
        deltas.append(delta)
        if bool(row.get("alpha_eligible_function")):
            eligible_deltas.append(delta)
    summary.update(
        {
            "paired_delta_vs_stage2_only": sum(deltas),
            "all_chunk_count": len(rows),
            "eligible_function_count": len(eligible_rows),
            "overall_hit_rate": (
                sum(int(bool(row.get("alpha_hit"))) for row in rows) / len(rows) if rows else 0.0
            ),
            "eligible_hit_rate": (
                sum(int(bool(row.get("alpha_hit"))) for row in eligible_rows) / len(eligible_rows)
                if eligible_rows
                else 0.0
            ),
            "eligible_positive_doc_rate": (
                sum(int(delta > 0) for delta in eligible_deltas) / len(eligible_deltas)
                if eligible_deltas
                else 0.0
            ),
            "eligible_mean_delta": (
                sum(eligible_deltas) / len(eligible_deltas) if eligible_deltas else 0.0
            ),
            "eligible_effective_saved": sum(int(row["effective_saved"]) for row in eligible_rows),
            "renamed_count_total": sum(int(row.get("alpha_renamed_count", 0) or 0) for row in rows),
            "variant_status": _first_nonempty(rows, "variant_status"),
            "train_allowed": any(bool(str(row.get("train_allowed", "")).lower() == "true" or row.get("train_allowed") is True) for row in rows),
            "train_block_reason": _first_nonempty(rows, "train_block_reason"),
            "ast_equivalence_rate": (
                sum(int(bool(row.get("ast_equivalent"))) for row in rows) / len(rows) if rows else 0.0
            ),
        }
    )
    return summary


def variant_training_policy(variant: str) -> tuple[str, bool, str]:
    if variant in {"stage2_alpha", "stage2_alpha_stage1_static"}:
        return "candidate", True, ""
    if variant in {"raw", "stage2_only"}:
        return "diagnostic_only", False, "baseline_only"
    if variant in {"stage2_alpha_stage1_tokenizer_opt", "stage1_only", "stage1_2"}:
        return "deprecated", False, "dynamic_stage1_prompt_codebook"
    if variant in {"stage3_only", "stage2_3", "stage1_2_3"}:
        return "deprecated", False, "old_stage3_dynamic_alias"
    return "diagnostic_only", False, "not_frozen_candidate"


def default_alpha_metadata() -> dict[str, object]:
    return {
        "alpha_applied": False,
        "alpha_renamed_count": 0,
        "alpha_raw_tokens": 0,
        "alpha_tokens": 0,
        "alpha_delta_tokens": 0,
        "alpha_guardrail_triggered": False,
        "alpha_rollback_reason": "",
        "alpha_skipped_reason": "",
        "alpha_ast_ok": False,
        "alpha_compile_ok": False,
        "alpha_public_signature_preserved": False,
    }


def _first_nonempty(rows: list[dict[str, object]], key: str) -> str:
    for row in rows:
        value = row.get(key, "")
        if value not in {"", None}:
            return str(value)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
