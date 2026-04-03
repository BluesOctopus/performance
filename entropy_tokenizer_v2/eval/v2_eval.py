"""Three-stage compress + aggregate token metrics (per file and corpus)."""

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

from tqdm.auto import tqdm

from . import bootstrap_v2

bootstrap_v2.ensure()

from config import (
    CACHE_DIR,
    EVAL_DATASET,
    EVAL_NUM_SAMPLES,
    EVAL_TOKENIZERS,
    HF_DISK_DATASET_FALLBACK,
    HF_TOKEN,
    RESULTS_DIR,
    STAGE2_DEFAULT_MODE,
    STAGE2_DEFAULT_PROFILE,
    VOCAB_COST_MODE,
)
from repo_miner import RepoConfig, _encode, _load_tokenizer, _vocab_size
from pipeline import CompressionBreakdown, apply_pipeline, _stage1_vocab_intro
from placeholder_accounting import compute_vocab_intro_cost, dedupe_vocab_entries
from token_scorer import (
    build_stage3_vocab_entries_from_used_placeholders,
    collect_used_stage3_placeholders,
)


def apply_v2_compression(
    source: str,
    repo_config: RepoConfig,
    tokenizer,
    tok_type: str,
    count_fn=None,
    stage2_profile: str | None = None,
    stage2_mode: str | None = None,
) -> tuple[str, CompressionBreakdown]:
    return apply_pipeline(
        source,
        repo_config,
        tokenizer,
        tok_type,
        count_fn=count_fn,
        stage2_profile=stage2_profile,
        stage2_mode=stage2_mode,
    )


def _entropy(token_counts: Counter) -> float:
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for cnt in token_counts.values():
        if cnt > 0:
            p = cnt / total
            h -= p * math.log2(p)
    return h


def _bpb(total_tokens: int, vocab_size: int, total_bytes: int) -> float:
    if total_bytes == 0:
        return 0.0
    return total_tokens * math.log2(vocab_size) / total_bytes


def load_eval_samples(num_samples: int = EVAL_NUM_SAMPLES) -> list[str]:
    ds_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", EVAL_DATASET)
    cache_path = CACHE_DIR / f"eval_samples_{ds_tag}_{num_samples}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return samples[:num_samples]

    if HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    try:
        from datasets import load_dataset
        ds = load_dataset(EVAL_DATASET, split="train",
                          streaming=True, token=HF_TOKEN or None)
        samples = []
        for ex in ds:
            samples.append(ex["content"])
            if len(samples) >= num_samples:
                break
    except Exception:
        from datasets import load_from_disk
        ds = load_from_disk(str(HF_DISK_DATASET_FALLBACK))
        samples = ds["content"][:num_samples]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    return samples


@dataclass
class EvalResult:
    tokenizer_key:       str
    stage2_profile:      str
    stage2_mode:         str
    n_files:             int
    baseline_tokens:     int
    syntax_tokens:       int
    cleaning_tokens:     int
    syntax_saved:        int
    cleaning_saved:      int
    replacement_saved:   int
    total_saved:         int
    sequence_final_tokens: int
    sequence_reduction_pct: float
    stage1_vocab_intro_tokens: int
    stage2_vocab_intro_tokens: int
    stage3_vocab_intro_tokens: int
    final_vocab_intro_tokens: int
    effective_total_tokens: int
    effective_total_saved: int
    effective_total_reduction_pct: float
    syntax_pct:          float
    cleaning_pct:        float
    replacement_pct:     float
    baseline_bpb:        float
    final_bpb:           float
    baseline_entropy:    float
    V0:                  int
    k_star_syntax:       int
    n_replacement_words: int
    stage3_vocab_intro_tokens_raw: int = 0
    stage3_vocab_intro_tokens_dedup: int = 0
    effective_total_tokens_dedup: int = 0
    effective_total_reduction_pct_dedup: float = 0.0
    stage3_backend: str = "legacy"
    stage3_selected_units: int = 0
    stage3_selected_units_exact: int = 0
    stage3_selected_units_semantic: int = 0
    stage3_used_units_exact: int = 0
    stage3_used_units_semantic: int = 0
    stage3_fields: str = ""
    stage3_dictionary_coverage: float = 0.0
    stage3_expected_gain: float = 0.0
    stage3_component_saved: int = 0
    stage3_used_entries: int = 0
    stage3_used_entries_variable: int = 0
    stage3_used_entries_attribute: int = 0
    stage3_used_entries_string: int = 0
    stage3_cost_model: str = ""
    stage3_plan_a_profile: str = ""
    stage3_assignment_by_field_json: str = ""
    stage3_vocab_scope: str = ""
    stage3_vocab_scope_detail: str = ""
    stage3_ab_a_candidates: int = 0
    stage3_ab_a_selected: int = 0
    stage3_ab_a_used_entries: int = 0
    stage3_ab_a_intro_tokens: int = 0
    stage3_ab_a_sequence_saved: int = 0
    stage3_ab_a_effective_net_saving: int = 0
    stage3_ab_a_protected_name_count: int = 0
    stage3_ab_a_min_occ_reject_count: int = 0
    stage3_ab_a_net_gain_reject_count: int = 0
    stage3_ab_a_route_reject_count: int = 0
    stage3_ab_a_reject_reason_json: str = ""
    stage3_ab_b_candidates: int = 0
    stage3_ab_b_cluster_count: int = 0
    stage3_ab_b_used_clusters: int = 0
    stage3_ab_b_intro_tokens: int = 0
    stage3_ab_b_sequence_saved: int = 0
    stage3_ab_b_effective_net_saving: int = 0
    stage3_ab_b_fallback_count: int = 0
    stage3_ab_b_intro_not_worth_count: int = 0
    stage3_ab_b_avg_similarity: float = 0.0
    stage3_ab_b_risk_reject_count: int = 0
    stage3_ab_b_route_reject_count: int = 0
    stage3_ab_b_reject_reason_json: str = ""
    stage3_ab_fallback_count: int = 0
    stage3_ab_mode: str = ""
    stage3_ab_similarity_kind: str = ""
    hybrid_ab_stage1_override_used: bool = False
    hybrid_ab_stage2_override_used: bool = False
    stage2_resolution_source: str = ""


def evaluate(
    sources: list[str],
    repo_config: RepoConfig,
    tokenizer_key: str,
    tokenizer_cfg: dict,
    stage2_profile: str | None = None,
    stage2_mode: str | None = None,
) -> EvalResult:
    tokenizer, tok_type = _load_tokenizer(tokenizer_key, tokenizer_cfg)
    V0 = _vocab_size(tokenizer, tok_type)
    count_fn = None

    total_bytes = sum(len(s.encode("utf-8")) for s in sources)

    backend = getattr(repo_config, "stage3_backend", "legacy")
    from pipeline import resolve_stage2_for_pipeline

    s2_eff_profile, s2_eff_mode, s2_src = resolve_stage2_for_pipeline(
        repo_config, stage2_profile, stage2_mode
    )
    hybrid_ab_s1 = bool(getattr(repo_config, "hybrid_ab_stage1_override_used", False))
    hybrid_ab_s2 = backend == "hybrid_ab" and s2_src == "hybrid_ab_default"
    plan_books = None
    plan_esc = getattr(repo_config, "stage3_escape_prefix", "__L__")
    if backend == "plan_a":
        from repo_miner import load_plan_a_codebooks

        plan_books = load_plan_a_codebooks(repo_config)

    agg = CompressionBreakdown(
        baseline_tokens=0, after_syntax=0,
        after_cleaning=0, after_replacement=0,
    )
    baseline_token_counts: Counter = Counter()
    legacy_stage3_used: list[str] = []
    legacy_seen: set[str] = set()
    plan_a_used_union: set[tuple[str, str]] = set()
    ab_vocab_entries: list[dict] = []
    ab_sum: Counter = Counter()
    ab_similarity_weighted_sum = 0.0
    ab_similarity_weight = 0
    ab_mode_seen = ""
    ab_similarity_kind_seen = ""
    ab_a_reject_reasons: Counter = Counter()
    ab_b_reject_reasons: Counter = Counter()

    for src in tqdm(sources, desc=f"  [{tokenizer_key}] compressing", leave=False):
        compressed, fr = apply_v2_compression(
            src,
            repo_config,
            tokenizer,
            tok_type,
            count_fn,
            stage2_profile=stage2_profile,
            stage2_mode=stage2_mode,
        )

        for tok_id in _encode(tokenizer, tok_type, src):
            baseline_token_counts[tok_id] += 1

        agg.baseline_tokens   += fr.baseline_tokens
        agg.after_syntax      += fr.after_syntax
        agg.after_cleaning    += fr.after_cleaning
        agg.after_replacement += fr.after_replacement

        if backend == "legacy" and repo_config.replacement_map:
            for ph in collect_used_stage3_placeholders(compressed, repo_config.replacement_map):
                if ph not in legacy_seen:
                    legacy_seen.add(ph)
                    legacy_stage3_used.append(ph)
        elif backend == "plan_a" and plan_books:
            from stage3.literal_codec.pipeline.source_codec import (
                extract_used_plan_a_entries,
            )

            plan_a_used_union |= extract_used_plan_a_entries(compressed, plan_books, plan_esc)
        elif backend == "hybrid_ab":
            meta = getattr(fr, "stage3_metrics", {}) or {}
            if not ab_mode_seen:
                ab_mode_seen = str(meta.get("stage3_ab_mode", "") or "")
            if not ab_similarity_kind_seen:
                ab_similarity_kind_seen = str(meta.get("stage3_ab_similarity_kind", "") or "")
            for k in (
                "stage3_ab_a_candidates",
                "stage3_ab_a_selected",
                "stage3_ab_a_used_entries",
                "stage3_ab_a_used_entries_variable",
                "stage3_ab_a_used_entries_attribute",
                "stage3_ab_a_used_entries_string",
                "stage3_ab_a_intro_tokens",
                "stage3_ab_a_sequence_saved",
                "stage3_ab_a_effective_net_saving",
                "stage3_ab_a_protected_name_count",
                "stage3_ab_a_min_occ_reject_count",
                "stage3_ab_a_net_gain_reject_count",
                "stage3_ab_b_candidates",
                "stage3_ab_b_cluster_count",
                "stage3_ab_b_used_clusters",
                "stage3_ab_b_intro_tokens",
                "stage3_ab_b_sequence_saved",
                "stage3_ab_b_effective_net_saving",
                "stage3_ab_b_fallback_count",
                "stage3_ab_b_risk_reject_count",
                "stage3_ab_b_intro_not_worth_count",
            ):
                ab_sum[k] += int(meta.get(k, 0))
            for rk, rv in (meta.get("stage3_ab_a_reject_reason_counts", {}) or {}).items():
                ab_a_reject_reasons[str(rk)] += int(rv)
            for rk, rv in (meta.get("stage3_ab_b_reject_reason_counts", {}) or {}).items():
                ab_b_reject_reasons[str(rk)] += int(rv)
            sim = float(meta.get("stage3_ab_b_avg_similarity", 0.0))
            used_cluster = int(meta.get("stage3_ab_b_used_clusters", 0))
            ab_similarity_weighted_sum += sim * used_cluster
            ab_similarity_weight += used_cluster
            ab_vocab_entries.extend(meta.get("stage3_ab_vocab_entries", []) or [])

    B = agg.baseline_tokens
    F_seq = agg.after_replacement
    F = max(1, F_seq)

    sm = getattr(repo_config, "stage3_plan_a_summary", None) or {}
    stage3_plan_a_assignments = int(sm.get("stage3_plan_a_assignments_count", 0))
    stage3_fields = ",".join(sm.get("stage3_plan_a_fields", []) or [])
    stage3_dictionary_coverage = float(sm.get("stage3_plan_a_dictionary_coverage_mean", 0.0))
    stage3_expected_gain = float(sm.get("stage3_plan_a_total_expected_gain_sum", 0.0))
    stage3_cost_model = str(sm.get("stage3_plan_a_cost_model", "") or "")
    stage3_plan_a_profile = str(sm.get("stage3_plan_a_profile", "") or "")
    stage3_vocab_scope = "corpus_once" if backend in {"legacy", "plan_a"} else "request_local_sum"
    stage3_vocab_scope_detail = (
        "legacy placeholders used-union / corpus-once"
        if backend == "legacy"
        else "plan_a used-codebook-union / corpus-once"
        if backend == "plan_a"
        else "hybrid_ab request-local entries (raw + dedup accounting)"
    )
    stage3_assignment_by_field_json = json.dumps(
        sm.get("stage3_plan_a_assignment_by_field", {}), ensure_ascii=False
    )

    stage3_vocab_intro_tokens_raw = 0
    stage3_vocab_intro_tokens_dedup = 0
    if backend == "plan_a":
        from placeholder_accounting import build_used_plan_a_vocab_entries
        from repo_miner import load_plan_a_codebooks

        books = plan_books if plan_books else load_plan_a_codebooks(repo_config)
        entries = build_used_plan_a_vocab_entries(
            books,
            plan_a_used_union,
            escape_prefix=plan_esc,
        )
        stage3_vocab_intro_tokens = compute_vocab_intro_cost(
            entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
        stage3_vocab_intro_tokens_raw = stage3_vocab_intro_tokens
        stage3_vocab_intro_tokens_dedup = stage3_vocab_intro_tokens
    elif backend == "legacy":
        entries = build_stage3_vocab_entries_from_used_placeholders(legacy_stage3_used)
        stage3_vocab_intro_tokens = compute_vocab_intro_cost(
            entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
        stage3_vocab_intro_tokens_raw = stage3_vocab_intro_tokens
        stage3_vocab_intro_tokens_dedup = stage3_vocab_intro_tokens
    else:
        entries_raw = ab_vocab_entries
        entries_dedup = dedupe_vocab_entries(entries_raw, key_fields=("token", "definition"))
        stage3_vocab_intro_tokens_raw = compute_vocab_intro_cost(
            entries_raw,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
        stage3_vocab_intro_tokens_dedup = compute_vocab_intro_cost(
            entries_dedup,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
        stage3_vocab_intro_tokens = stage3_vocab_intro_tokens_raw

    used_v = used_a = used_s = 0
    if backend == "plan_a":
        for fld, _code in plan_a_used_union:
            if fld == "variable":
                used_v += 1
            elif fld == "attribute":
                used_a += 1
            elif fld == "string":
                used_s += 1

    stage3_component_saved = agg.replacement_saved
    stage3_selected_units = 0
    stage3_selected_units_exact = 0
    stage3_selected_units_semantic = 0
    stage3_used_units_exact = 0
    stage3_used_units_semantic = 0
    if backend == "hybrid_ab":
        used_v = int(ab_sum.get("stage3_ab_a_used_entries_variable", 0))
        used_a = int(ab_sum.get("stage3_ab_a_used_entries_attribute", 0))
        # string used = A-string + B-used-clusters
        used_s = int(ab_sum.get("stage3_ab_a_used_entries_string", 0) + ab_sum.get("stage3_ab_b_used_clusters", 0))
        stage3_selected_units_exact = int(ab_sum.get("stage3_ab_a_selected", 0))
        stage3_selected_units_semantic = int(ab_sum.get("stage3_ab_b_used_clusters", 0))
        stage3_selected_units = stage3_selected_units_exact + stage3_selected_units_semantic
        stage3_used_units_exact = int(ab_sum.get("stage3_ab_a_used_entries", 0))
        stage3_used_units_semantic = int(ab_sum.get("stage3_ab_b_used_clusters", 0))
        stage3_component_saved = int(
            ab_sum.get("stage3_ab_a_sequence_saved", 0)
            + ab_sum.get("stage3_ab_b_sequence_saved", 0)
        )
    elif backend == "plan_a":
        stage3_selected_units = stage3_plan_a_assignments
        stage3_selected_units_exact = stage3_plan_a_assignments
        stage3_selected_units_semantic = 0
        stage3_used_units_exact = len(plan_a_used_union)
        stage3_used_units_semantic = 0
    else:
        stage3_selected_units = len(getattr(repo_config, "replacement_map", {}) or {})
        stage3_selected_units_exact = stage3_selected_units
        stage3_selected_units_semantic = 0
        stage3_used_units_exact = len(legacy_stage3_used)
        stage3_used_units_semantic = 0

    s1_intro = _stage1_vocab_intro(repo_config, tokenizer, tok_type)
    s2_intro = 0
    final_vocab_intro = s1_intro + s2_intro + stage3_vocab_intro_tokens
    final_vocab_intro_dedup = s1_intro + s2_intro + stage3_vocab_intro_tokens_dedup
    effective_total = F_seq + final_vocab_intro
    effective_total_dedup = F_seq + final_vocab_intro_dedup
    effective_saved = B - effective_total
    seq_red = (1.0 - F_seq / B) * 100.0 if B else 0.0
    eff_red = (1.0 - effective_total / B) * 100.0 if B else 0.0
    eff_red_dedup = (1.0 - effective_total_dedup / B) * 100.0 if B else 0.0

    return EvalResult(
        tokenizer_key       = tokenizer_key,
        stage2_profile      = s2_eff_profile,
        stage2_mode         = s2_eff_mode,
        n_files             = len(sources),
        baseline_tokens     = B,
        syntax_tokens       = agg.after_syntax,
        cleaning_tokens     = agg.after_cleaning,
        syntax_saved        = agg.syntax_saved,
        cleaning_saved      = agg.cleaning_saved,
        replacement_saved   = agg.replacement_saved,
        total_saved         = agg.total_saved,
        sequence_final_tokens=F_seq,
        sequence_reduction_pct=seq_red,
        stage1_vocab_intro_tokens=s1_intro,
        stage2_vocab_intro_tokens=s2_intro,
        stage3_vocab_intro_tokens=stage3_vocab_intro_tokens,
        final_vocab_intro_tokens=final_vocab_intro,
        effective_total_tokens=effective_total,
        effective_total_saved=effective_saved,
        effective_total_reduction_pct=eff_red,
        syntax_pct          = (agg.syntax_saved   / B) * 100.0 if B else 0.0,
        cleaning_pct        = (agg.cleaning_saved / B) * 100.0 if B else 0.0,
        replacement_pct     = (agg.replacement_saved / B) * 100.0 if B else 0.0,
        baseline_bpb        = _bpb(B, V0, total_bytes),
        final_bpb           = _bpb(F, V0, total_bytes),
        baseline_entropy    = _entropy(baseline_token_counts),
        V0                  = V0,
        k_star_syntax       = len(repo_config.selected_skeletons),
        n_replacement_words = len(repo_config.replacement_map),
        stage3_vocab_intro_tokens_raw=stage3_vocab_intro_tokens_raw,
        stage3_vocab_intro_tokens_dedup=stage3_vocab_intro_tokens_dedup,
        effective_total_tokens_dedup=effective_total_dedup,
        effective_total_reduction_pct_dedup=eff_red_dedup,
        stage3_backend      = backend,
        stage3_selected_units=stage3_selected_units,
        stage3_selected_units_exact=stage3_selected_units_exact,
        stage3_selected_units_semantic=stage3_selected_units_semantic,
        stage3_used_units_exact=stage3_used_units_exact,
        stage3_used_units_semantic=stage3_used_units_semantic,
        stage3_fields       = stage3_fields,
        stage3_dictionary_coverage = stage3_dictionary_coverage,
        stage3_expected_gain = stage3_expected_gain,
        stage3_component_saved = stage3_component_saved,
        stage3_used_entries=(
            len(plan_a_used_union)
            if backend == "plan_a"
            else int(ab_sum.get("stage3_ab_a_used_entries", 0) + ab_sum.get("stage3_ab_b_used_clusters", 0))
            if backend == "hybrid_ab"
            else 0
        ),
        stage3_used_entries_variable=used_v,
        stage3_used_entries_attribute=used_a,
        stage3_used_entries_string=used_s,
        stage3_cost_model=stage3_cost_model,
        stage3_plan_a_profile=stage3_plan_a_profile,
        stage3_assignment_by_field_json=stage3_assignment_by_field_json,
        stage3_vocab_scope=stage3_vocab_scope,
        stage3_vocab_scope_detail=stage3_vocab_scope_detail,
        stage3_ab_a_candidates=int(ab_sum.get("stage3_ab_a_candidates", 0)),
        stage3_ab_a_selected=int(ab_sum.get("stage3_ab_a_selected", 0)),
        stage3_ab_a_used_entries=int(ab_sum.get("stage3_ab_a_used_entries", 0)),
        stage3_ab_a_intro_tokens=int(ab_sum.get("stage3_ab_a_intro_tokens", 0)),
        stage3_ab_a_sequence_saved=int(ab_sum.get("stage3_ab_a_sequence_saved", 0)),
        stage3_ab_a_effective_net_saving=int(ab_sum.get("stage3_ab_a_effective_net_saving", 0)),
        stage3_ab_a_protected_name_count=int(ab_sum.get("stage3_ab_a_protected_name_count", 0)),
        stage3_ab_a_min_occ_reject_count=int(ab_sum.get("stage3_ab_a_min_occ_reject_count", 0)),
        stage3_ab_a_net_gain_reject_count=int(ab_sum.get("stage3_ab_a_net_gain_reject_count", 0)),
        stage3_ab_a_route_reject_count=int(sum(ab_a_reject_reasons.values())),
        stage3_ab_a_reject_reason_json=json.dumps(dict(ab_a_reject_reasons), ensure_ascii=False),
        stage3_ab_b_candidates=int(ab_sum.get("stage3_ab_b_candidates", 0)),
        stage3_ab_b_cluster_count=int(ab_sum.get("stage3_ab_b_cluster_count", 0)),
        stage3_ab_b_used_clusters=int(ab_sum.get("stage3_ab_b_used_clusters", 0)),
        stage3_ab_b_intro_tokens=int(ab_sum.get("stage3_ab_b_intro_tokens", 0)),
        stage3_ab_b_sequence_saved=int(ab_sum.get("stage3_ab_b_sequence_saved", 0)),
        stage3_ab_b_effective_net_saving=int(ab_sum.get("stage3_ab_b_effective_net_saving", 0)),
        stage3_ab_b_fallback_count=int(ab_sum.get("stage3_ab_b_fallback_count", 0)),
        stage3_ab_b_intro_not_worth_count=int(ab_sum.get("stage3_ab_b_intro_not_worth_count", 0)),
        stage3_ab_b_avg_similarity=(
            ab_similarity_weighted_sum / ab_similarity_weight if ab_similarity_weight else 0.0
        ),
        stage3_ab_b_risk_reject_count=int(ab_sum.get("stage3_ab_b_risk_reject_count", 0)),
        stage3_ab_b_route_reject_count=int(sum(ab_b_reject_reasons.values())),
        stage3_ab_b_reject_reason_json=json.dumps(dict(ab_b_reject_reasons), ensure_ascii=False),
        stage3_ab_fallback_count=int(ab_sum.get("stage3_ab_b_fallback_count", 0)),
        stage3_ab_mode=ab_mode_seen or str((getattr(repo_config, "stage3_ab_summary", {}) or {}).get("stage3_ab_mode", "")),
        stage3_ab_similarity_kind=ab_similarity_kind_seen or str(
            (getattr(repo_config, "stage3_ab_summary", {}) or {}).get("stage3_ab_similarity_kind", "")
        ),
        hybrid_ab_stage1_override_used=hybrid_ab_s1,
        hybrid_ab_stage2_override_used=hybrid_ab_s2,
        stage2_resolution_source=s2_src,
    )


def print_report(results: list[EvalResult]):
    w = 168
    print("\n" + "=" * w)
    print("  v2 compression — evaluation report")
    print("=" * w)

    hdr = (
        f"  {'Tokenizer':<20} {'Profile':<18} {'Mode':<9} {'Baseline':>10} "
        f"{'SeqFinal':>10} {'Seq%':>7} {'EffTotal':>10} {'Eff%':>7} "
        f"{'Vocab':>8} {'S1voc':>6} {'S3voc':>6} "
        f"{'Syn%':>6} {'Cln%':>6} {'Tok%':>6} "
        f"{'BPB_b':>8} {'BPB_f':>8} {'K*':>4} {'Nrp':>5} {'S3be':>8}"
    )
    print(hdr)
    print("-" * w)

    for r in results:
        print(
            f"  {r.tokenizer_key:<20} {r.stage2_profile:<18} {r.stage2_mode:<9} "
            f"{r.baseline_tokens:>10,} {r.sequence_final_tokens:>10,} "
            f"{r.sequence_reduction_pct:>6.1f}% {r.effective_total_tokens:>10,} "
            f"{r.effective_total_reduction_pct:>6.1f}% "
            f"{r.final_vocab_intro_tokens:>8,} {r.stage1_vocab_intro_tokens:>6,} "
            f"{r.stage3_vocab_intro_tokens:>6,} "
            f"{r.syntax_pct:>5.1f}% {r.cleaning_pct:>5.1f}% {r.replacement_pct:>5.1f}% "
            f"{r.baseline_bpb:>8.4f} {r.final_bpb:>8.4f} "
            f"{r.k_star_syntax:>4} {r.n_replacement_words:>5} "
            f"{r.stage3_backend:>8}"
        )

    print("=" * w)
    print(
        "  SeqFinal / Seq% = corpus sequence tokens only (placeholders as 1); "
        "EffTotal / Eff% = SeqFinal + corpus-once vocab intro (S1+S2+S3); "
        "Vocab = S1voc+S2voc+S3voc (S2voc=0 here); S3voc = Stage3 intro only; "
        "Syn%/Cln%/Tok% = stage1/2/3 sequence savings vs baseline; "
        "Nrp = legacy replacement_map size (0 for plan_a)."
    )


def save_results(
    results: list[EvalResult],
    repo_config_by_tok: dict,
    *,
    csv_name: str = "v2_compression_report.csv",
    detail_name: str = "v2_eval_detail.json",
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / csv_name
    fields = [f.name for f in EvalResult.__dataclass_fields__.values()] + [
        "final_tokens",
        "reduction_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # CSV compatibility aliases (not part of EvalResult schema).
            row["final_tokens"] = row["sequence_final_tokens"]
            row["reduction_pct"] = row["sequence_reduction_pct"]
            for k in (
                "reduction_pct",
                "sequence_reduction_pct",
                "effective_total_reduction_pct",
                "syntax_pct",
                "cleaning_pct",
                "replacement_pct",
                "baseline_bpb",
                "final_bpb",
                "baseline_entropy",
                "stage3_dictionary_coverage",
                "stage3_expected_gain",
                "stage3_ab_b_avg_similarity",
            ):
                row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"\n[eval] CSV saved → {csv_path}")

    detail_path = RESULTS_DIR / detail_name
    detail = {
        "results": [asdict(r) for r in results],
        "top_scores_by_tokenizer": {
            tok_key: cfg.scores_summary
            for tok_key, cfg in repo_config_by_tok.items()
        },
        "selected_skeletons_by_tokenizer": {
            tok_key: cfg.selected_skeletons
            for tok_key, cfg in repo_config_by_tok.items()
        },
        "stage3_plan_a_summary_by_tokenizer": {
            tok_key: getattr(cfg, "stage3_plan_a_summary", {})
            for tok_key, cfg in repo_config_by_tok.items()
        },
        "stage3_hybrid_ab_summary_by_tokenizer": {
            tok_key: getattr(cfg, "stage3_ab_summary", {})
            for tok_key, cfg in repo_config_by_tok.items()
        },
    }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"[eval] Detail JSON saved → {detail_path}")


def eval_mining_cache_name(
    num_samples: int,
    backend: str,
    stage2_profile: str | None,
    stage2_mode: str | None,
) -> str:
    """
    Corpus mining does not depend on Stage2, but historical caches embed a Stage2 tag.
    hybrid_ab + implicit Stage2 uses a dedicated tag so caches stay stable and distinct.
    """
    if stage2_profile is None and stage2_mode is None:
        if backend == "hybrid_ab":
            s2tag = "s2_hybrid_ab_auto"
        else:
            s2tag = f"{STAGE2_DEFAULT_PROFILE}_{STAGE2_DEFAULT_MODE}"
    else:
        p = stage2_profile if stage2_profile is not None else STAGE2_DEFAULT_PROFILE
        m = stage2_mode if stage2_mode is not None else STAGE2_DEFAULT_MODE
        s2tag = f"{p}_{m}"
    return f"eval_{num_samples}_{s2tag}"


def run_evaluation(
    tokenizer_keys: Optional[list[str]] = None,
    num_samples: int = EVAL_NUM_SAMPLES,
    verbose: bool = True,
    stage2_profile: str | None = None,
    stage2_mode: str | None = None,
    *,
    stage3_backend: str | None = None,
    output_csv: str = "v2_compression_report.csv",
    output_detail: str = "v2_eval_detail.json",
) -> list[EvalResult]:
    from config import STAGE3_BACKEND
    from repo_miner import mine_from_sources

    backend = (stage3_backend or STAGE3_BACKEND).strip().lower()
    if backend not in {"legacy", "plan_a", "hybrid_ab"}:
        backend = "legacy"

    if tokenizer_keys is None:
        tokenizer_keys = list(EVAL_TOKENIZERS.keys())

    if verbose:
        print(f"[eval] Loading {num_samples} evaluation samples ...")
    samples = load_eval_samples(num_samples)
    if verbose:
        print(f"[eval] Loaded {len(samples)} samples  "
              f"({sum(len(s.encode()) for s in samples):,} bytes total)")

    all_results: list[EvalResult] = []
    repo_config_by_tok: dict[str, RepoConfig] = {}

    for tok_key in tokenizer_keys:
        cfg = EVAL_TOKENIZERS.get(tok_key)
        if cfg is None:
            print(f"[eval] Unknown tokenizer '{tok_key}', skipping")
            continue

        print(f"\n{'='*60}\n  TOKENIZER: {tok_key}\n{'='*60}")

        repo_config = mine_from_sources(
            sources=samples,
            tokenizer_key=tok_key,
            tokenizer_cfg=cfg,
            cache_name=eval_mining_cache_name(num_samples, backend, stage2_profile, stage2_mode),
            cache=True,
            verbose=verbose,
            stage3_backend=backend,
        )
        repo_config_by_tok[tok_key] = repo_config

        result = evaluate(
            samples,
            repo_config,
            tok_key,
            cfg,
            stage2_profile=stage2_profile,
            stage2_mode=stage2_mode,
        )
        all_results.append(result)

        if verbose:
            print(
                f"  Sequence reduction: {result.sequence_reduction_pct:.1f}%  "
                f"(Syntax {result.syntax_pct:.1f}% + "
                f"Clean {result.cleaning_pct:.1f}% + "
                f"Token {result.replacement_pct:.1f}%)  |  "
                f"Effective-total reduction: {result.effective_total_reduction_pct:.1f}%  "
                f"(vocab intro {result.final_vocab_intro_tokens:,} tok)"
            )

    print_report(all_results)
    save_results(all_results, repo_config_by_tok, csv_name=output_csv, detail_name=output_detail)
    return all_results
