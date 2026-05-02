from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import resolve_hybrid_ab_settings
from marker_optimizer import MarkerScheme, build_legacy_marker_scheme, build_marker_scheme
from markers import (
    extract_line_marker,
    extract_stage1_markers,
    get_active_marker_scheme,
    make_syn_marker,
    set_active_marker_scheme,
)
from pipeline import (
    _stage3_encode_result,
    apply_stage1_stage2_adapted,
    apply_stage1_with_stats,
)
from placeholder_accounting import (
    compute_vocab_intro_cost,
    dedupe_vocab_entries,
    extract_unique_placeholders,
)
from repo_miner import RepoConfig
from stage2.cleaning import run_stage2_pre_safe
from stage2.config import build_stage2_execution_plan
from stage3.guardrail import GuardrailDecision, apply_effective_guardrail
from syntax_compressor import (
    SkeletonCandidate,
    build_candidate_pool,
    build_stage1_vocab_entry,
    collect_skeleton_occurrences,
    greedy_mdl_select,
    mine_skeletons,
    score_skeleton_candidate,
)
from tokenizer_utils import DEFAULT_TOKENIZER_NAME, count_tokens, resolve_tokenizer

CODEBOOK_COST_MODE = "serialized_definition"
DEFAULT_CODEBOOK_ACCOUNTING_MODE = "per_chunk"
DEFAULT_STAGE1_MARKER_SCHEME = "tokenizer_opt"


@dataclass(frozen=True)
class TokenLedger:
    tokenizer_name: str
    raw_tokens: int
    compressed_tokens: int
    codebook_tokens: int
    wrapper_tokens: int
    task_prompt_tokens: int
    effective_tokens: int
    effective_prompt_tokens: int
    gross_saved: int
    net_saved: int
    effective_saved: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1DecodeResult:
    decode_success: bool
    roundtrip_ok: bool
    ast_equivalent: bool
    decoded_text: str
    error_type: str


def resolve_encoder() -> Any:
    return resolve_tokenizer(DEFAULT_TOKENIZER_NAME)


def resolve_encoder_for_name(tokenizer_name: str) -> Any:
    return resolve_tokenizer(tokenizer_name)


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_stage_repo_config(
    chunk_texts: list[str],
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    stage1_marker_scheme: str = DEFAULT_STAGE1_MARKER_SCHEME,
    stage3_mode: str = "exact_only",
    enable_b: bool = False,
    stage3_min_occ: int = 2,
    stage3_min_gain: int = 1,
) -> RepoConfig:
    all_skeleton_counts = mine_skeletons(chunk_texts, min_freq=1)
    skeleton_counts = mine_skeletons(chunk_texts)
    n_baseline = sum(count_tokens(text, encoder=encoder, tok_type=tok_type) for text in chunk_texts)
    vocab_size = int(getattr(encoder, "n_vocab", 0) or 0)
    candidate_markers = resolve_stage1_marker_scheme(
        stage1_marker_scheme,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        count=len(skeleton_counts),
        avoid_texts=chunk_texts,
    )
    rejection_counts = classify_stage1_rejections(
        all_skeleton_counts,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        chunk_texts=chunk_texts,
        marker_scheme=candidate_markers,
    )
    candidates = build_candidate_pool(
        skeleton_counts,
        encoder,
        tok_type,
        chunk_texts,
        marker_tokens=candidate_markers.markers,
    )
    selected, _diag, stage1_total_net_saving = greedy_mdl_select(
        candidates,
        n_baseline,
        vocab_size,
        return_diagnostics=True,
    )
    selected_set = {candidate.skeleton for candidate in selected}
    stage1_candidate_stats = [_candidate_to_stats(candidate, candidate.skeleton in selected_set) for candidate in candidates]
    stage1_selected_stats = [row for row in stage1_candidate_stats if row["selected"]]

    stage3_summary = dict(resolve_hybrid_ab_settings("gpt4" if tok_type == "tiktoken" else "qwen"))
    stage3_summary["mode"] = stage3_mode
    stage3_summary["enable_b"] = bool(enable_b and stage3_mode == "hybrid")
    stage3_summary["a_min_occ"] = int(stage3_min_occ)
    stage3_summary["a_min_net_gain"] = int(stage3_min_gain)
    stage3_summary["enable_global_guardrail"] = False
    stage3_summary["enable_incremental_rollback"] = False

    return RepoConfig(
        selected_skeletons=[asdict(candidate) for candidate in selected],
        replacement_map={},
        scores_summary=[],
        stage1_candidate_stats=stage1_candidate_stats,
        stage1_selected_stats=stage1_selected_stats,
        stage1_total_net_saving=float(stage1_total_net_saving),
        n_sources=len(chunk_texts),
        N_baseline_tokens=n_baseline,
        V0=vocab_size,
        tokenizer_key=tokenizer_name,
        stage1_marker_scheme=normalize_stage1_marker_scheme(stage1_marker_scheme),
        stage1_marker_namespace=candidate_markers.namespace,
        stage1_marker_tokens=candidate_markers.markers[: len(selected)],
        stage1_marker_token_costs=candidate_markers.marker_token_costs[: len(selected)],
        stage1_all_skeleton_count=int(rejection_counts["all_skeleton_count"]),
        stage1_low_frequency_count=int(rejection_counts["low_frequency_count"]),
        stage1_rejected_no_gain_count=int(rejection_counts["rejected_no_gain_count"]),
        stage1_rejected_intro_cost_count=int(rejection_counts["rejected_intro_cost_count"]),
        stage1_rejected_marker_cost_count=int(rejection_counts["rejected_marker_cost_count"]),
        stage3_backend="hybrid_ab",
        stage3_ab_summary=stage3_summary,
    )


def normalize_stage2_profile(stage2_profile: str) -> str:
    lowered = stage2_profile.strip().lower()
    if lowered == "safe":
        return "safe"
    if lowered in {"aggressive", "aggressive_upper_bound"}:
        return "aggressive_upper_bound"
    raise ValueError(f"Unsupported stage2 profile: {stage2_profile!r}")


def normalize_stage1_marker_scheme(stage1_marker_scheme: str) -> str:
    lowered = stage1_marker_scheme.strip().lower()
    if lowered in {"legacy", "legacy_marker"}:
        return "legacy"
    if lowered in {"tokenizer_opt", "tokenizer_opt_marker"}:
        return "tokenizer_opt"
    raise ValueError(f"Unsupported Stage1 marker scheme: {stage1_marker_scheme!r}")


def resolve_stage1_marker_scheme(
    stage1_marker_scheme: str,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    count: int,
    avoid_texts: list[str] | None = None,
    namespace: str = "syn",
) -> MarkerScheme:
    normalized = normalize_stage1_marker_scheme(stage1_marker_scheme)
    if normalized == "legacy":
        return build_legacy_marker_scheme(tokenizer_name, encoder, tok_type, count)
    return build_marker_scheme(
        tokenizer_name,
        encoder,
        tok_type,
        namespace,
        count,
        avoid_texts=avoid_texts,
    )


def stage1_marker_scheme_from_repo_config(
    repo_config: RepoConfig,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
) -> MarkerScheme:
    skeleton_count = len(repo_config.skeleton_candidates()) if hasattr(repo_config, "skeleton_candidates") else 0
    markers = list(getattr(repo_config, "stage1_marker_tokens", []) or [])
    costs = list(getattr(repo_config, "stage1_marker_token_costs", []) or [])
    if markers:
        if len(costs) != len(markers):
            costs = [count_tokens(marker, encoder=encoder, tok_type=tok_type) for marker in markers]
        return MarkerScheme(
            tokenizer_name=tokenizer_name,
                namespace=str(getattr(repo_config, "stage1_marker_namespace", "repo") or "repo"),
                markers=markers,
                marker_token_costs=costs,
            )
    return build_legacy_marker_scheme(tokenizer_name, encoder, tok_type, skeleton_count)


def classify_stage1_rejections(
    all_skeleton_counts,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    chunk_texts: list[str],
    marker_scheme: MarkerScheme,
) -> dict[str, int]:
    del tokenizer_name
    occ_min = 2
    occurrences_map = collect_skeleton_occurrences(chunk_texts, set(all_skeleton_counts.keys()))
    counts = {
        "all_skeleton_count": len(all_skeleton_counts),
        "low_frequency_count": 0,
        "rejected_no_gain_count": 0,
        "rejected_intro_cost_count": 0,
        "rejected_marker_cost_count": 0,
    }
    for index, skeleton in enumerate(all_skeleton_counts.keys()):
        occurrences = occurrences_map.get(skeleton, [])
        if len(occurrences) < occ_min:
            counts["low_frequency_count"] += 1
            continue
        marker = marker_scheme.marker(index) if index < len(marker_scheme.markers) else make_syn_marker(index)
        stats = score_skeleton_candidate(
            skeleton,
            occurrences,
            marker,
            encoder,
            tok_type,
        )
        total_net_saving = int(stats.get("total_net_saving", 0))
        effective_total_net_saving = int(stats.get("effective_total_net_saving", total_net_saving))
        marker_cost = int(stats.get("marker_cost", 0))
        occurrences_count = int(stats.get("occurrences", 0))
        vocab_intro_tokens = int(stats.get("vocab_intro_tokens", 0))
        if total_net_saving <= 0:
            counts["rejected_no_gain_count"] += 1
        elif marker_cost * max(occurrences_count, 1) >= total_net_saving:
            counts["rejected_marker_cost_count"] += 1
        elif effective_total_net_saving <= 0 or vocab_intro_tokens >= total_net_saving:
            counts["rejected_intro_cost_count"] += 1
    return counts


def apply_stage2_only(text: str, *, stage2_profile: str, path: str = "<chunk>") -> str:
    plan = build_stage2_execution_plan(normalize_stage2_profile(stage2_profile))
    cleaned, _stats = run_stage2_pre_safe(text, plan.pre_cfg, path=path)
    return cleaned


def apply_stage1_only(
    text: str,
    repo_config: RepoConfig,
    *,
    encoder: Any,
    tok_type: str,
) -> tuple[str, dict[str, dict[str, int]]]:
    scheme = stage1_marker_scheme_from_repo_config(
        repo_config,
        tokenizer_name=str(getattr(repo_config, "tokenizer_key", "")) or DEFAULT_TOKENIZER_NAME,
        encoder=encoder,
        tok_type=tok_type,
    )
    set_active_marker_scheme(scheme)
    try:
        return apply_stage1_with_stats(text, repo_config, encoder, tok_type)
    finally:
        set_active_marker_scheme(None)


def apply_stage1_stage2(
    text: str,
    repo_config: RepoConfig,
    *,
    encoder: Any,
    tok_type: str,
    stage2_profile: str,
    path: str,
) -> dict[str, Any]:
    scheme = stage1_marker_scheme_from_repo_config(
        repo_config,
        tokenizer_name=str(getattr(repo_config, "tokenizer_key", "")) or DEFAULT_TOKENIZER_NAME,
        encoder=encoder,
        tok_type=tok_type,
    )
    set_active_marker_scheme(scheme)
    try:
        return apply_stage1_stage2_adapted(
            text,
            repo_config,
            stage2_profile=normalize_stage2_profile(stage2_profile),
            tokenizer=encoder,
            tok_type=tok_type,
            path=path,
        )
    finally:
        set_active_marker_scheme(None)


def apply_stage3(text: str, repo_config: RepoConfig, *, encoder: Any, tok_type: str) -> dict[str, Any]:
    result, _backend = _stage3_encode_result(text, repo_config, encoder, tok_type)
    vocab_entries = dedupe_vocab_entries(list(result.vocab_entries or []))
    metrics = dict(result.metrics or {})
    selected_count = int(metrics.get("stage3_ab_a_selected", 0)) + int(
        metrics.get("stage3_ab_b_used_clusters", 0)
    )
    candidate_count = int(metrics.get("stage3_ab_a_candidates", 0)) + int(
        metrics.get("stage3_ab_b_candidates", 0)
    )
    projected_gain = int(metrics.get("stage3_ab_a_effective_net_saving", 0)) + int(
        metrics.get("stage3_ab_b_effective_net_saving", 0)
    )
    return {
        "text": result.encoded_text,
        "vocab_entries": vocab_entries,
        "metrics": metrics,
        "stage3_triggered": selected_count > 0,
        "stage3_candidate_count": candidate_count,
        "stage3_selected_count": selected_count,
        "projected_gain": projected_gain,
    }


def stage1_vocab_entries_for_text(text: str, repo_config: RepoConfig) -> list[dict[str, Any]]:
    scheme = MarkerScheme(
        tokenizer_name=str(getattr(repo_config, "tokenizer_key", "")) or DEFAULT_TOKENIZER_NAME,
        namespace=str(getattr(repo_config, "stage1_marker_namespace", "repo") or "repo"),
        markers=list(getattr(repo_config, "stage1_marker_tokens", []) or []),
        marker_token_costs=list(getattr(repo_config, "stage1_marker_token_costs", []) or []),
    )
    marker_map = {
        (candidate.marker_text or make_syn_marker(index)): candidate.skeleton
        for index, candidate in enumerate(repo_config.skeleton_candidates())
    }
    entries: list[dict[str, Any]] = []
    for placeholder in extract_stage1_markers(text, scheme=scheme if scheme.markers else None):
        skeleton = marker_map.get(placeholder)
        if skeleton is None:
            continue
        entries.append(build_stage1_vocab_entry(placeholder, skeleton))
    return dedupe_vocab_entries(entries)


def stage1_marker_metrics(
    text: str,
    repo_config: RepoConfig,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
) -> dict[str, Any]:
    skeleton_count = len(repo_config.skeleton_candidates()) if hasattr(repo_config, "skeleton_candidates") else 0
    active_scheme = stage1_marker_scheme_from_repo_config(
        repo_config,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
    )
    markers = extract_stage1_markers(text, scheme=active_scheme if active_scheme.markers else None)
    if not markers:
        return {
            "stage1_marker_scheme": str(getattr(repo_config, "stage1_marker_scheme", "legacy") or "legacy"),
            "marker_tokens_total": 0,
            "avg_marker_token_cost": 0.0,
            "legacy_marker_tokens_total": 0,
            "tokenizer_opt_marker_tokens_total": 0,
            "marker_saved_vs_legacy": 0,
        }

    legacy_scheme = build_legacy_marker_scheme(tokenizer_name, encoder, tok_type, skeleton_count)
    tokenizer_opt_scheme = resolve_stage1_marker_scheme(
        "tokenizer_opt",
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        count=skeleton_count,
        avoid_texts=None,
    )
    marker_to_index = {marker: index for index, marker in enumerate(active_scheme.markers)}
    marker_tokens_total = 0
    legacy_marker_tokens_total = 0
    tokenizer_opt_marker_tokens_total = 0
    for marker in markers:
        index = marker_to_index.get(marker)
        if index is None:
            continue
        current_cost = active_scheme.marker_cost(index)
        legacy_cost = legacy_scheme.marker_cost(index)
        optimized_cost = tokenizer_opt_scheme.marker_cost(index)
        marker_tokens_total += current_cost
        legacy_marker_tokens_total += legacy_cost
        tokenizer_opt_marker_tokens_total += optimized_cost

    avg_marker_token_cost = marker_tokens_total / len(markers) if markers else 0.0
    return {
        "stage1_marker_scheme": str(getattr(repo_config, "stage1_marker_scheme", "legacy") or "legacy"),
        "marker_tokens_total": marker_tokens_total,
        "avg_marker_token_cost": avg_marker_token_cost,
        "legacy_marker_tokens_total": legacy_marker_tokens_total,
        "tokenizer_opt_marker_tokens_total": tokenizer_opt_marker_tokens_total,
        "marker_saved_vs_legacy": legacy_marker_tokens_total - marker_tokens_total,
    }


def build_token_ledger(
    raw_text: str,
    compressed_text: str,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    codebook_entries: list[dict[str, Any]] | None = None,
    wrapper_text: str = "",
    task_prompt_text: str = "",
) -> TokenLedger:
    raw_tokens = count_tokens(raw_text, encoder=encoder, tok_type=tok_type)
    compressed_tokens = count_tokens(compressed_text, encoder=encoder, tok_type=tok_type)
    wrapper_tokens = count_tokens(wrapper_text, encoder=encoder, tok_type=tok_type) if wrapper_text else 0
    task_prompt_tokens = (
        count_tokens(task_prompt_text, encoder=encoder, tok_type=tok_type) if task_prompt_text else 0
    )
    codebook_tokens = compute_vocab_intro_cost(
        dedupe_vocab_entries(codebook_entries or []),
        mode=CODEBOOK_COST_MODE,
        tokenizer=encoder,
        tok_type=tok_type,
    )
    effective_tokens = compressed_tokens + codebook_tokens + wrapper_tokens
    effective_prompt_tokens = effective_tokens + task_prompt_tokens
    return TokenLedger(
        tokenizer_name=tokenizer_name,
        raw_tokens=raw_tokens,
        compressed_tokens=compressed_tokens,
        codebook_tokens=codebook_tokens,
        wrapper_tokens=wrapper_tokens,
        task_prompt_tokens=task_prompt_tokens,
        effective_tokens=effective_tokens,
        effective_prompt_tokens=effective_prompt_tokens,
        gross_saved=raw_tokens - compressed_tokens,
        net_saved=raw_tokens - effective_tokens,
        effective_saved=raw_tokens - effective_prompt_tokens,
    )


def summarize_variant(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    stage2_profile: str,
    codebook_accounting_mode: str,
) -> dict[str, Any]:
    if not rows:
        return {"variant": variant}
    if codebook_accounting_mode == "global_once":
        codebook_entries = []
        for row in rows:
            codebook_entries.extend(list(row.get("_codebook_entries", [])))
        codebook_tokens = compute_vocab_intro_cost(
            dedupe_vocab_entries(codebook_entries),
            mode=CODEBOOK_COST_MODE,
            tokenizer=encoder,
            tok_type=tok_type,
        )
    elif codebook_accounting_mode == "per_chunk":
        codebook_tokens = sum(int(row["codebook_tokens"]) for row in rows)
    else:
        raise ValueError(f"Unsupported codebook accounting mode: {codebook_accounting_mode!r}")
    n_rows = len(rows)
    compressed_tokens = sum(int(row["compressed_tokens"]) for row in rows)
    wrapper_tokens = sum(int(row["wrapper_tokens"]) for row in rows)
    task_prompt_tokens = sum(int(row["task_prompt_tokens"]) for row in rows)
    effective_tokens = compressed_tokens + codebook_tokens + wrapper_tokens
    summary = {
        "variant": variant,
        "tokenizer_name": tokenizer_name,
        "stage2_profile": stage2_profile,
        "codebook_accounting_mode": codebook_accounting_mode,
        "raw_tokens": sum(int(row["raw_tokens"]) for row in rows),
        "compressed_tokens": compressed_tokens,
        "codebook_tokens": codebook_tokens,
        "wrapper_tokens": wrapper_tokens,
        "effective_tokens": effective_tokens,
        "gross_saved": sum(int(row["gross_saved"]) for row in rows),
        "net_saved": sum(int(row["raw_tokens"]) for row in rows) - effective_tokens,
        "effective_saved": sum(int(row["raw_tokens"]) for row in rows) - (effective_tokens + task_prompt_tokens),
        "stage1_marker_scheme": _first_value(rows, "stage1_marker_scheme", ""),
        "marker_tokens_total": sum(int(row.get("marker_tokens_total", 0)) for row in rows),
        "avg_marker_token_cost": _mean_numeric(rows, "avg_marker_token_cost"),
        "legacy_marker_tokens_total": sum(int(row.get("legacy_marker_tokens_total", 0)) for row in rows),
        "tokenizer_opt_marker_tokens_total": sum(
            int(row.get("tokenizer_opt_marker_tokens_total", 0)) for row in rows
        ),
        "marker_saved_vs_legacy": sum(int(row.get("marker_saved_vs_legacy", 0)) for row in rows),
        "stage1_hit_rate": _mean_bool(rows, "stage1_hit"),
        "stage3_trigger_rate": _mean_bool(rows, "stage3_triggered"),
        "roundtrip_success_rate": _mean_bool(rows, "roundtrip_ok"),
        "ast_equivalence_rate": _mean_bool(rows, "ast_equivalent"),
        "decode_success_rate": _mean_bool(rows, "decode_success"),
        "chunk_count": n_rows,
    }
    return summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    fieldnames = list(public_rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)


def decode_stage1_text(text: str, repo_config: RepoConfig, *, original_text: str) -> Stage1DecodeResult:
    scheme = MarkerScheme(
        tokenizer_name=str(getattr(repo_config, "tokenizer_key", "")) or DEFAULT_TOKENIZER_NAME,
        namespace=str(getattr(repo_config, "stage1_marker_namespace", "repo") or "repo"),
        markers=list(getattr(repo_config, "stage1_marker_tokens", []) or []),
        marker_token_costs=list(getattr(repo_config, "stage1_marker_token_costs", []) or []),
    )
    marker_map = {
        (candidate.marker_text or make_syn_marker(index)): candidate.skeleton
        for index, candidate in enumerate(repo_config.skeleton_candidates())
    }
    decoded_lines: list[str] = []
    error_type = ""
    for line in text.splitlines():
        marker = extract_line_marker(line, scheme=scheme if scheme.markers else None)
        if marker is None:
            decoded_lines.append(line)
            continue
        stripped = line.strip()
        parts = stripped.split(maxsplit=1)
        payload = parts[1] if len(parts) > 1 else ""
        skeleton = marker_map.get(marker)
        if skeleton is None:
            return Stage1DecodeResult(False, False, False, text, "unknown_marker")
        slot_indices = sorted({int(match.group(1)) for match in re.finditer(r"\{(\d+)\}", skeleton)})
        if len(slot_indices) > 1:
            return Stage1DecodeResult(False, False, False, text, "ambiguous_slot_boundaries")
        if not slot_indices:
            decoded_lines.append(skeleton)
            continue
        if slot_indices != [0]:
            return Stage1DecodeResult(False, False, False, text, "non_contiguous_slots")
        decoded_lines.append(skeleton.replace("{0}", payload))

    decoded_text = "\n".join(decoded_lines)
    decode_success = True
    roundtrip_ok = decoded_text == original_text
    ast_equivalent = ast_equivalent_text(original_text, decoded_text)
    if not roundtrip_ok and not ast_equivalent:
        error_type = "ast_mismatch_after_decode"
    return Stage1DecodeResult(
        decode_success=decode_success,
        roundtrip_ok=roundtrip_ok,
        ast_equivalent=ast_equivalent,
        decoded_text=decoded_text,
        error_type=error_type,
    )


def ast_equivalent_text(left: str, right: str) -> bool:
    try:
        left_tree = ast.parse(left)
        right_tree = ast.parse(right)
    except SyntaxError:
        return False
    return ast.dump(left_tree, include_attributes=False) == ast.dump(
        right_tree, include_attributes=False
    )


def with_guardrail(
    *,
    tokenizer_name: str,
    tok_type: str,
    raw_reference_text: str,
    baseline_text: str,
    baseline_entries: list[dict[str, Any]],
    candidate_text: str,
    candidate_entries: list[dict[str, Any]],
    encoder: Any,
    rollback_label: str,
) -> tuple[TokenLedger, GuardrailDecision]:
    baseline_ledger = build_token_ledger(
        raw_reference_text,
        baseline_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=baseline_entries,
    )
    candidate_ledger = build_token_ledger(
        raw_reference_text,
        candidate_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        codebook_entries=candidate_entries,
    )
    decision = apply_effective_guardrail(
        baseline_effective_tokens=baseline_ledger.effective_tokens,
        candidate_effective_tokens=candidate_ledger.effective_tokens,
        rollback_label=rollback_label,
    )
    final_ledger = baseline_ledger if decision.should_rollback else candidate_ledger
    return final_ledger, decision


def _candidate_to_stats(candidate: SkeletonCandidate, selected: bool) -> dict[str, Any]:
    return {
        "skeleton": candidate.skeleton,
        "occurrences": candidate.frequency,
        "avg_baseline_cost": candidate.avg_baseline_cost,
        "avg_compressed_cost": candidate.avg_compressed_cost,
        "avg_slot_cost": candidate.avg_slot_cost,
        "marker_cost": candidate.marker_cost,
        "avg_net_saving": candidate.avg_net_saving,
        "total_net_saving": candidate.total_net_saving,
        "vocab_intro_tokens": candidate.vocab_intro_tokens,
        "effective_total_net_saving": candidate.effective_total_net_saving,
        "avg_sequence_net_saving": candidate.avg_sequence_net_saving,
        "selected": selected,
    }


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    true_count = 0
    for row in rows:
        value = row.get(key, False)
        if isinstance(value, str):
            value = value.lower() in {"1", "true", "yes"}
        true_count += int(bool(value))
    return true_count / len(rows)


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)


def _first_value(rows: list[dict[str, Any]], key: str, default: Any) -> Any:
    for row in rows:
        if key in row:
            return row[key]
    return default


def stage3_decode_status(
    *,
    repo_config: RepoConfig,
    encoded_text: str,
    original_text: str,
) -> dict[str, Any]:
    del repo_config, encoded_text, original_text
    return {
        "decode_success": False,
        "roundtrip_ok": False,
        "ast_equivalent": False,
        "error_type": "stage3_decoder_missing",
    }
