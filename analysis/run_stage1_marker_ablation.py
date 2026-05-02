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
    apply_stage2_pipeline,
    build_stage_repo_config,
    build_token_ledger,
    decode_stage1_text,
    load_chunks,
    resolve_encoder_for_name,
    stable_text_hash,
    stage1_marker_metrics,
    stage1_vocab_entries_for_text,
    summarize_variant,
    write_csv,
)
from placeholder_accounting import compute_vocab_intro_cost, dedupe_vocab_entries

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
    parser.add_argument(
        "--stage1-codebook-mode",
        default="per_chunk",
        choices=("per_chunk", "global_once", "pretrained_static_vocab"),
        help="Primary Stage1 codebook accounting mode for detail rows",
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
                args.stage1_codebook_mode,
            )
            for chunk in chunks
        ]
        variant_rows_map[variant] = variant_rows
        detail_rows.extend(variant_rows)

    summary_rows = [
        build_summary_row(
            variant_rows_map[variant],
            variant=variant,
            repo_config=repo_config_for_variant(variant, repo_configs),
            tokenizer_name=resolved.tokenizer_name,
            encoder=resolved.encoder,
            tok_type=resolved.tok_type,
            stage2_profile=args.stage2_profile,
            stage1_codebook_mode=args.stage1_codebook_mode,
        )
        for variant in VARIANTS
    ]

    stage2_summary = next((row for row in summary_rows if row["variant"] == "stage2_only"), None)
    legacy_summary = next((row for row in summary_rows if row["variant"] == "stage1_legacy"), None)
    tokenizer_opt_summary = next((row for row in summary_rows if row["variant"] == "stage1_tokenizer_opt"), None)
    legacy_2_summary = next((row for row in summary_rows if row["variant"] == "stage1_legacy_2"), None)
    tokenizer_opt_2_summary = next((row for row in summary_rows if row["variant"] == "stage1_tokenizer_opt_2"), None)

    for row in summary_rows:
        row["tokenizer_opt_beats_legacy"] = bool(
            tokenizer_opt_summary
            and legacy_summary
            and int(tokenizer_opt_summary["pretrained_static_vocab_effective_saved"])
            > int(legacy_summary["pretrained_static_vocab_effective_saved"])
        )
        row["tokenizer_opt_stage1_2_beats_stage2_only"] = bool(
            tokenizer_opt_2_summary
            and stage2_summary
            and int(tokenizer_opt_2_summary["pretrained_static_vocab_effective_saved"])
            > int(stage2_summary["pretrained_static_vocab_effective_saved"])
        )
        row["tokenizer_opt_stage1_2_delta_vs_stage2_only"] = (
            int(tokenizer_opt_2_summary["pretrained_static_vocab_effective_saved"])
            - int(stage2_summary["pretrained_static_vocab_effective_saved"])
            if tokenizer_opt_2_summary and stage2_summary
            else 0
        )
        row["tokenizer_opt_delta_vs_legacy"] = (
            int(tokenizer_opt_summary["pretrained_static_vocab_effective_saved"])
            - int(legacy_summary["pretrained_static_vocab_effective_saved"])
            if tokenizer_opt_summary and legacy_summary
            else 0
        )
        row["tokenizer_opt_2_delta_vs_legacy_2"] = (
            int(tokenizer_opt_2_summary["pretrained_static_vocab_effective_saved"])
            - int(legacy_2_summary["pretrained_static_vocab_effective_saved"])
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
    stage1_codebook_mode: str,
) -> dict[str, object]:
    original_text = str(chunk["chunk_text"])
    source_id = str(chunk["source_id"])
    baseline_stage2 = apply_stage2_pipeline(original_text, stage2_profile=stage2_profile, path=source_id)
    baseline_stage2_text = str(baseline_stage2["stage2_post_text"])

    repo_config = repo_config_for_variant(variant, repo_configs)
    stage1_hit = False
    roundtrip_ok = False
    ast_equivalent = False
    decode_success = False
    error_type = ""
    codebook_entries: list[dict[str, object]] = []
    stage1_text = ""
    final_text = original_text

    if variant == "raw":
        final_text = original_text
    elif variant == "stage2_only":
        final_text = baseline_stage2_text
    elif variant == "stage1_legacy":
        stage1_text, _ = apply_stage1_only(original_text, repo_config, encoder=encoder, tok_type=tok_type)
        final_text = stage1_text
    elif variant == "stage1_tokenizer_opt":
        stage1_text, _ = apply_stage1_only(original_text, repo_config, encoder=encoder, tok_type=tok_type)
        final_text = stage1_text
    elif variant == "stage1_legacy_2":
        adapted = apply_stage1_stage2(
            original_text,
            repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        stage1_text = str(adapted["stage1_text"])
        final_text = str(adapted["stage2_post_text"])
    elif variant == "stage1_tokenizer_opt_2":
        adapted = apply_stage1_stage2(
            original_text,
            repo_config,
            encoder=encoder,
            tok_type=tok_type,
            stage2_profile=stage2_profile,
            path=source_id,
        )
        stage1_text = str(adapted["stage1_text"])
        final_text = str(adapted["stage2_post_text"])
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    if stage1_text:
        baseline_for_decode = baseline_stage2["stage2_pre_text"] if variant.endswith("_2") else original_text
        stage1_hit = stage1_text != baseline_for_decode
        decode_result = decode_stage1_text(stage1_text, repo_config, original_text=baseline_for_decode)
        roundtrip_ok = decode_result.roundtrip_ok
        ast_equivalent = decode_result.ast_equivalent
        decode_success = decode_result.decode_success
        error_type = decode_result.error_type
        codebook_entries = stage1_vocab_entries_for_text(stage1_text, repo_config)

    per_chunk_ledger = build_token_ledger(
        original_text,
        final_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=codebook_entries,
    )
    selected_ledger = ledger_for_mode(
        stage1_codebook_mode,
        raw_text=original_text,
        compressed_text=final_text,
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
    stage2_post_text_hash = stable_text_hash(final_text)
    baseline_hash = stable_text_hash(baseline_stage2_text)
    stage2_path_equal = variant.endswith("_2") and stage2_post_text_hash == baseline_hash
    warning_flag = variant.endswith("_2") and (not stage1_hit) and (not stage2_path_equal)
    repo_diag = stage1_repo_diagnostics(repo_config, [dict(stage1_hit=stage1_hit, variant=variant)])
    return {
        "variant": variant,
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "symbol_type": chunk["symbol_type"],
        "symbol_name": chunk["symbol_name"],
        "tokenizer_name": tokenizer_name,
        "stage2_profile": stage2_profile,
        "stage1_codebook_mode": stage1_codebook_mode,
        "stage1_marker_scheme": str(getattr(repo_config, "stage1_marker_scheme", "legacy") or "legacy"),
        "raw_tokens": selected_ledger.raw_tokens,
        "compressed_tokens": selected_ledger.compressed_tokens,
        "codebook_tokens": selected_ledger.codebook_tokens,
        "wrapper_tokens": selected_ledger.wrapper_tokens,
        "task_prompt_tokens": selected_ledger.task_prompt_tokens,
        "effective_tokens": selected_ledger.effective_tokens,
        "effective_prompt_tokens": selected_ledger.effective_prompt_tokens,
        "gross_saved": selected_ledger.gross_saved,
        "net_saved": selected_ledger.net_saved,
        "effective_saved": selected_ledger.effective_saved,
        "per_chunk_codebook_tokens": per_chunk_ledger.codebook_tokens,
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
        "stage2_only_text_hash": baseline_hash,
        "stage1_2_baseline_stage2_text_hash": baseline_hash if variant.endswith("_2") else "",
        "stage1_text_hash": stable_text_hash(stage1_text) if stage1_text else "",
        "stage2_post_text_hash": stage2_post_text_hash,
        "stage2_path_equal": stage2_path_equal,
        "stage1_noop_but_stage2_diff_warning": warning_flag,
        "_codebook_entries": codebook_entries,
    }


def ledger_for_mode(
    mode: str,
    *,
    raw_text: str,
    compressed_text: str,
    tokenizer_name: str,
    encoder,
    tok_type: str,
    codebook_entries: list[dict[str, object]],
):
    entries = [] if mode in {"global_once", "pretrained_static_vocab"} else codebook_entries
    ledger = build_token_ledger(
        raw_text,
        compressed_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=entries,
    )
    if mode == "pretrained_static_vocab":
        return build_token_ledger(
            raw_text,
            compressed_text,
            tokenizer_name=tokenizer_name,
            encoder=encoder,
            tok_type=tok_type,
            codebook_entries=[],
        )
    return ledger


def repo_config_for_variant(variant: str, repo_configs: dict[str, object]):
    return repo_configs["tokenizer_opt"] if "tokenizer_opt" in variant else repo_configs["legacy"]


def build_summary_row(
    rows: list[dict[str, object]],
    *,
    variant: str,
    repo_config,
    tokenizer_name: str,
    encoder,
    tok_type: str,
    stage2_profile: str,
    stage1_codebook_mode: str,
) -> dict[str, object]:
    summary = summarize_variant(
        rows,
        variant=variant,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        stage2_profile=stage2_profile,
        codebook_accounting_mode=stage1_codebook_mode,
    )
    summary.update(stage1_repo_diagnostics(repo_config, rows))
    per_chunk_codebook_tokens = sum(int(row.get("per_chunk_codebook_tokens", 0) or 0) for row in rows)
    global_once_codebook_tokens = compute_vocab_intro_cost(
        dedupe_vocab_entries(
            [entry for row in rows for entry in list(row.get("_codebook_entries", []))]
        ),
        mode="serialized_definition",
        tokenizer=encoder,
        tok_type=tok_type,
    )
    raw_tokens_total = sum(int(row["raw_tokens"]) for row in rows)
    compressed_tokens_total = sum(int(row["compressed_tokens"]) for row in rows)
    wrapper_tokens_total = sum(int(row["wrapper_tokens"]) for row in rows)
    task_prompt_tokens_total = sum(int(row["task_prompt_tokens"]) for row in rows)
    break_even_codebook_tokens = raw_tokens_total - compressed_tokens_total - wrapper_tokens_total - task_prompt_tokens_total
    summary.update(
        {
            "per_chunk_effective_saved": raw_tokens_total
            - (compressed_tokens_total + wrapper_tokens_total + task_prompt_tokens_total + per_chunk_codebook_tokens),
            "global_once_effective_saved": raw_tokens_total
            - (compressed_tokens_total + wrapper_tokens_total + task_prompt_tokens_total + global_once_codebook_tokens),
            "pretrained_static_vocab_effective_saved": raw_tokens_total
            - (compressed_tokens_total + wrapper_tokens_total + task_prompt_tokens_total),
            "break_even_codebook_tokens": break_even_codebook_tokens,
            "codebook_overhead_ratio": (
                float(summary["codebook_tokens"]) / break_even_codebook_tokens if break_even_codebook_tokens > 0 else 0.0
            ),
            "stage1_ablation_invalid": any(bool(row.get("stage1_noop_but_stage2_diff_warning")) for row in rows),
            "warning_type": (
                "stage2_path_mismatch_when_stage1_noop"
                if any(bool(row.get("stage1_noop_but_stage2_diff_warning")) for row in rows)
                else ""
            ),
        }
    )
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
    if bool(summary.get("stage1_ablation_invalid")):
        return "stage1_ablation_invalid; stage2 path mismatch when Stage1 noop"
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
