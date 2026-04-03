"""Core compression pipeline (Stage1 -> Stage2 -> Stage3)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from config import (
    STAGE2_DEFAULT_MODE,
    STAGE2_DEFAULT_PROFILE,
    VOCAB_COST_MODE,
)
from marker_count import count_augmented
from markers import RE_ALL_MARKERS, get_syn_line_spans, make_syn_marker
from placeholder_accounting import compute_vocab_intro_cost
from stage2.cleaning import (
    run_stage2_post_surface,
    run_stage2_pre_safe,
    stage2_clean_skip_syn,
)
from stage2.config import (
    STAGE2_LAYOUT_EXPERIMENTAL_ORDER_LABEL,
    build_stage2_config,
    build_stage2_execution_plan,
)
from stage2.layout_encoding import run_stage2_post_layout_encode
from syntax_compressor import build_stage1_vocab_entry, compress_source_syntax
from token_scorer import (
    apply_token_replacement_with_protected_spans,
    build_stage3_vocab_entries_from_used_placeholders,
    collect_used_stage3_placeholders,
)


@dataclass
class CompressionBreakdown:
    """
    Per-file **sequence-only** token counts (placeholders count as 1 each).

    * ``after_replacement`` — final compressed **sequence** tokens for this file (no vocab intro).
    * ``stage{1,2,3}_vocab_intro_tokens`` — corpus-once intro costs **replicated** on each row for
      convenience; eval must **not** sum these across files (use one shot + corpus-wide Stage3 union).
    * ``final_effective_total_tokens`` — sequence + vocab for this row only (per-file; not corpus).
    """

    baseline_tokens: int
    after_syntax: int
    after_cleaning: int
    after_replacement: int
    # One-shot vocab introduction (same numeric value on every file for this repo_config).
    stage1_vocab_intro_tokens: int = 0
    stage2_vocab_intro_tokens: int = 0
    stage3_vocab_intro_tokens: int = 0

    @property
    def syntax_saved(self) -> int:
        return self.baseline_tokens - self.after_syntax

    @property
    def cleaning_saved(self) -> int:
        return self.after_syntax - self.after_cleaning

    @property
    def replacement_saved(self) -> int:
        return self.after_cleaning - self.after_replacement

    @property
    def total_saved(self) -> int:
        return self.baseline_tokens - self.after_replacement

    @property
    def final_sequence_only_tokens(self) -> int:
        return self.after_replacement

    @property
    def final_vocab_intro_tokens(self) -> int:
        return (
            self.stage1_vocab_intro_tokens
            + self.stage2_vocab_intro_tokens
            + self.stage3_vocab_intro_tokens
        )

    @property
    def final_effective_total_tokens(self) -> int:
        return self.after_replacement + self.final_vocab_intro_tokens


def _count_with_ops(text: str, tokenizer, tok_type: str) -> int:
    return count_augmented(text, tokenizer, tok_type, pattern=RE_ALL_MARKERS)


def _stage1_vocab_intro(repo_config, tokenizer, tok_type: str) -> int:
    cands = repo_config.skeleton_candidates()
    if not cands:
        return 0
    entries = [
        build_stage1_vocab_entry(make_syn_marker(i), c.skeleton)
        for i, c in enumerate(cands)
    ]
    return compute_vocab_intro_cost(
        entries,
        mode=VOCAB_COST_MODE,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )


def _stage3_vocab_intro(repo_config, after_s3_text: str, tokenizer, tok_type: str) -> int:
    backend = getattr(repo_config, "stage3_backend", "legacy")
    if backend == "plan_a":
        from placeholder_accounting import build_used_plan_a_vocab_entries

        from repo_miner import load_plan_a_codebooks

        from stage3.literal_codec.pipeline.source_codec import (
            extract_used_plan_a_entries,
        )

        codebooks = load_plan_a_codebooks(repo_config)
        if not codebooks:
            return 0
        esc = getattr(repo_config, "stage3_escape_prefix", "__L__")
        used = extract_used_plan_a_entries(after_s3_text, codebooks, esc)
        entries = build_used_plan_a_vocab_entries(codebooks, used, escape_prefix=esc)
        return compute_vocab_intro_cost(
            entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
    if backend == "hybrid_ab":
        meta = getattr(repo_config, "_stage3_hybrid_last_meta", {}) or {}
        entries = meta.get("stage3_ab_vocab_entries", []) or []
        if not entries:
            return 0
        return compute_vocab_intro_cost(
            entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
    rmap = repo_config.replacement_map
    if not rmap:
        return 0
    used = collect_used_stage3_placeholders(after_s3_text, rmap)
    entries = build_stage3_vocab_entries_from_used_placeholders(used)
    return compute_vocab_intro_cost(
        entries,
        mode=VOCAB_COST_MODE,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )


def apply_stage1(source: str, repo_config) -> str:
    return compress_source_syntax(source, repo_config.skeleton_candidates())


def apply_stage1_with_stats(
    source: str,
    repo_config,
    tokenizer,
    tok_type: str,
) -> tuple[str, dict[str, dict]]:
    return compress_source_syntax(
        source,
        repo_config.skeleton_candidates(),
        tokenizer=tokenizer,
        tok_type=tok_type,
        prune_nonpositive=True,
        return_stats=True,
    )


def _parse_ok(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError):
        return False
    except Exception:
        return False


def apply_stage1_stage2_adapted(
    source: str,
    repo_config,
    *,
    stage2_profile: str,
    tokenizer,
    tok_type: str,
    path: str | None = None,
) -> dict[str, Any]:
    """
    Adapted main line: ``stage2_pre_safe -> stage1 -> stage2_post_surface``.

    * *source* should be parseable Python (e.g. lossless-cleaned corpus text).
    * *stage2_profile*: ``safe`` or ``aggressive_upper_bound``.
    """
    plan = build_stage2_execution_plan(stage2_profile)
    pre_text, pre_stats = run_stage2_pre_safe(source, plan.pre_cfg, path=path)
    stage1_text, _s1_detail = apply_stage1_with_stats(
        pre_text, repo_config, tokenizer, tok_type
    )
    post_text, post_stats = run_stage2_post_surface(
        stage1_text, plan.post_cfg, path=path
    )

    baseline_sequence_tokens = _count_with_ops(source, tokenizer, tok_type)
    stage2_pre_sequence_tokens = _count_with_ops(pre_text, tokenizer, tok_type)
    stage1_sequence_tokens = _count_with_ops(stage1_text, tokenizer, tok_type)
    final_sequence_tokens = _count_with_ops(post_text, tokenizer, tok_type)

    s1_intro = _stage1_vocab_intro(repo_config, tokenizer, tok_type)
    final_effective_total_tokens = final_sequence_tokens + s1_intro

    cands = repo_config.skeleton_candidates()
    selected_skeletons = [x.skeleton for x in cands]
    stage1_vocab_tokens = [make_syn_marker(i) for i in range(len(cands))]

    return {
        "original_text": source,
        "stage2_pre_text": pre_text,
        "stage1_text": stage1_text,
        "stage2_post_text": post_text,
        "baseline_sequence_tokens": baseline_sequence_tokens,
        "stage2_pre_sequence_tokens": stage2_pre_sequence_tokens,
        "stage1_sequence_tokens": stage1_sequence_tokens,
        "final_sequence_tokens": final_sequence_tokens,
        "stage1_vocab_intro_tokens": s1_intro,
        "final_effective_total_tokens": final_effective_total_tokens,
        "baseline_parse_success": _parse_ok(source),
        "stage2_pre_parse_success": _parse_ok(pre_text),
        "stage1_parse_success": _parse_ok(stage1_text),
        "final_parse_success": _parse_ok(post_text),
        "stage2_pre_stats": pre_stats,
        "stage2_post_stats": post_stats,
        "selected_skeletons": selected_skeletons,
        "stage1_vocab_tokens": stage1_vocab_tokens,
        "stage2_order": plan.order_label,
        "stage2_profile": stage2_profile,
    }


def apply_stage1_stage2_layout_safe_experimental(
    source: str,
    repo_config,
    *,
    tokenizer,
    tok_type: str,
    path: str | None = None,
) -> dict[str, Any]:
    """
    Experimental: ``stage2_pre_safe -> stage1 -> post_surface (safe) -> layout encode``.

    Does not strip indentation; layout tokens are reversible on the post-surface text.
    """
    plan = build_stage2_execution_plan("safe")
    pre_text, pre_stats = run_stage2_pre_safe(source, plan.pre_cfg, path=path)
    stage1_text, _s1_detail = apply_stage1_with_stats(
        pre_text, repo_config, tokenizer, tok_type
    )
    post_surface_text, post_stats = run_stage2_post_surface(
        stage1_text, plan.post_cfg, path=path
    )
    post_text, layout_meta = run_stage2_post_layout_encode(
        post_surface_text, path=path
    )

    baseline_sequence_tokens = _count_with_ops(source, tokenizer, tok_type)
    stage2_pre_sequence_tokens = _count_with_ops(pre_text, tokenizer, tok_type)
    stage1_sequence_tokens = _count_with_ops(stage1_text, tokenizer, tok_type)
    post_surface_sequence_tokens = _count_with_ops(
        post_surface_text, tokenizer, tok_type
    )
    final_sequence_tokens = _count_with_ops(post_text, tokenizer, tok_type)

    s1_intro = _stage1_vocab_intro(repo_config, tokenizer, tok_type)
    final_effective_total_tokens = final_sequence_tokens + s1_intro

    cands = repo_config.skeleton_candidates()
    selected_skeletons = [x.skeleton for x in cands]
    stage1_vocab_tokens = [make_syn_marker(i) for i in range(len(cands))]

    return {
        "original_text": source,
        "stage2_pre_text": pre_text,
        "stage1_text": stage1_text,
        "stage2_post_surface_text": post_surface_text,
        "stage2_post_text": post_text,
        "baseline_sequence_tokens": baseline_sequence_tokens,
        "stage2_pre_sequence_tokens": stage2_pre_sequence_tokens,
        "stage1_sequence_tokens": stage1_sequence_tokens,
        "stage2_post_surface_sequence_tokens": post_surface_sequence_tokens,
        "final_sequence_tokens": final_sequence_tokens,
        "stage1_vocab_intro_tokens": s1_intro,
        "final_effective_total_tokens": final_effective_total_tokens,
        "baseline_parse_success": _parse_ok(source),
        "stage2_pre_parse_success": _parse_ok(pre_text),
        "stage1_parse_success": _parse_ok(stage1_text),
        "final_parse_success": _parse_ok(post_surface_text),
        "stage2_pre_stats": pre_stats,
        "stage2_post_stats": post_stats,
        "selected_skeletons": selected_skeletons,
        "stage1_vocab_tokens": stage1_vocab_tokens,
        "stage2_order": STAGE2_LAYOUT_EXPERIMENTAL_ORDER_LABEL,
        "stage2_profile": "layout_safe_experimental",
        "layout_meta": layout_meta,
    }


def apply_stage2(
    text: str,
    *,
    profile: str = STAGE2_DEFAULT_PROFILE,
    mode: str = STAGE2_DEFAULT_MODE,
) -> str:
    stage2_cfg = build_stage2_config(profile=profile, mode=mode)
    return stage2_clean_skip_syn(
        text,
        stage2_cfg.cleaning,
        mode=stage2_cfg.mode,
        drop_empty_cleaned_lines=False,
    )


def apply_stage3(text: str, repo_config, tokenizer=None, tok_type: str | None = None) -> str:
    backend = getattr(repo_config, "stage3_backend", "legacy")
    if backend == "plan_a":
        from stage3.literal_codec.pipeline.source_codec import (
            encode_python_source_plan_a,
        )

        from repo_miner import load_plan_a_codebooks

        books = load_plan_a_codebooks(repo_config)
        if not books:
            return text
        esc = getattr(repo_config, "stage3_escape_prefix", "__L__")
        return encode_python_source_plan_a(text, books, escape_prefix=esc)
    if backend == "hybrid_ab":
        if tokenizer is None or not tok_type:
            return text
        from stage3.hybrid_ab import (
            HybridABConfig,
            encode_stage3_hybrid_ab,
            summary_dict,
        )

        cfg_raw = dict(getattr(repo_config, "stage3_ab_summary", {}) or {})
        if not cfg_raw:
            # Strictly avoid hidden defaults in pipeline; skip Stage3 safely.
            setattr(
                repo_config,
                "_stage3_hybrid_last_meta",
                {
                    "stage3_ab_mode": "exact_only",
                    "stage3_ab_similarity_kind": "lexical_bow_cosine",
                    "stage3_ab_b_mode": "disabled",
                    "stage3_ab_runtime_warning": "missing stage3_ab_summary; stage3 skipped",
                    "stage3_ab_vocab_entries": [],
                },
            )
            return text
        mode = str(cfg_raw.get("mode", "exact_only")).strip().lower()
        if mode not in {"exact_only", "hybrid"}:
            mode = "exact_only"
        conf = HybridABConfig(
            mode=mode,
            free_text_min_chars=int(cfg_raw["free_text_min_chars"]),
            free_text_min_words=int(cfg_raw["free_text_min_words"]),
            key_like_patterns=tuple(cfg_raw.get("key_like_patterns", []) or ()),
            a_min_occ=int(cfg_raw["a_min_occ"]),
            a_min_net_gain=int(cfg_raw["a_min_net_gain"]),
            a_alias_style=str(cfg_raw["a_alias_style"]),
            b_similarity_threshold=float(cfg_raw["b_similarity_threshold"]),
            b_risk_threshold=float(cfg_raw["b_risk_threshold"]),
            b_min_cluster_size=int(cfg_raw["b_min_cluster_size"]),
        )
        if mode != "hybrid" or not bool(cfg_raw.get("enable_b", False)):
            conf.mode = "exact_only"
        res = encode_stage3_hybrid_ab(text, tokenizer=tokenizer, tok_type=tok_type, cfg=conf)
        setattr(repo_config, "_stage3_hybrid_last_meta", summary_dict(res))
        return res.encoded_text
    rmap = repo_config.replacement_map
    if not rmap:
        return text
    protected_spans = get_syn_line_spans(text)
    return apply_token_replacement_with_protected_spans(text, rmap, protected_spans)


def apply_pipeline(
    source: str,
    repo_config,
    tokenizer,
    tok_type: str,
    count_fn=None,
    stage2_profile: str = STAGE2_DEFAULT_PROFILE,
    stage2_mode: str = STAGE2_DEFAULT_MODE,
) -> tuple[str, CompressionBreakdown]:
    if count_fn is None:
        def count_fn_local(text: str) -> int:
            return _count_with_ops(text, tokenizer, tok_type)

        count_fn = count_fn_local

    baseline_tokens = count_fn(source)
    after_s1 = apply_stage1_with_stats(source, repo_config, tokenizer, tok_type)[0]
    after_s1_tokens = count_fn(after_s1)

    after_s2 = apply_stage2(after_s1, profile=stage2_profile, mode=stage2_mode)
    after_s2_tokens = count_fn(after_s2)

    try:
        after_s3 = apply_stage3(after_s2, repo_config, tokenizer=tokenizer, tok_type=tok_type)
    except TypeError:
        # Backward-compatible for tests/mocks patching apply_stage3(text, repo_config).
        after_s3 = apply_stage3(after_s2, repo_config)
    after_s3_tokens = count_fn(after_s3)

    s1_intro = _stage1_vocab_intro(repo_config, tokenizer, tok_type)
    s3_intro = _stage3_vocab_intro(repo_config, after_s3, tokenizer, tok_type)

    breakdown = CompressionBreakdown(
        baseline_tokens=baseline_tokens,
        after_syntax=after_s1_tokens,
        after_cleaning=after_s2_tokens,
        after_replacement=after_s3_tokens,
        stage1_vocab_intro_tokens=s1_intro,
        stage2_vocab_intro_tokens=0,
        stage3_vocab_intro_tokens=s3_intro,
    )
    return after_s3, breakdown
