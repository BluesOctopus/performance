"""Core compression pipeline (Stage1 -> Stage2 -> Stage3)."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field, replace
from typing import Any

from config import (
    STAGE2_DEFAULT_MODE,
    STAGE2_DEFAULT_PROFILE,
    VOCAB_COST_MODE,
)
from marker_count import count_augmented
from markers import RE_ALL_MARKERS, make_syn_marker
from placeholder_accounting import compute_vocab_intro_cost
from stage3.backends import get_stage3_backend
from stage3.backends.base import Stage3EncodeResult
from lossy_cleaner import CleaningStats
from stage2.cleaning import (
    run_stage2_post_surface,
    run_stage2_pre_safe,
    stage2_clean_skip_syn_and_stats,
)
from stage2.config import (
    STAGE2_LAYOUT_EXPERIMENTAL_ORDER_LABEL,
    build_stage2_config,
    build_stage2_execution_plan,
)
from stage2.layout_encoding import run_stage2_post_layout_encode
from syntax_compressor import build_stage1_vocab_entry, compress_source_syntax


def resolve_stage2_for_pipeline(
    repo_config,
    stage2_profile: str | None,
    stage2_mode: str | None,
) -> tuple[str, str, str]:
    """
    Pick Stage2 profile/mode for ``apply_pipeline``.

    Priority:
    1. Explicit *stage2_profile* and/or *stage2_mode* (missing half filled from global defaults).
    2. ``hybrid_ab`` backend → hybrid_ab-specific defaults (env ``ET_STAGE2_HYBRID_AB_*``).
    3. Otherwise global ``STAGE2_DEFAULT_*``.

    Returns (profile, mode, resolution_source) where *resolution_source* is
    ``explicit`` | ``hybrid_ab_default`` | ``global_default``.
    """
    if stage2_profile is not None or stage2_mode is not None:
        p = stage2_profile if stage2_profile is not None else STAGE2_DEFAULT_PROFILE
        m = stage2_mode if stage2_mode is not None else STAGE2_DEFAULT_MODE
        return p, m, "explicit"
    backend = getattr(repo_config, "stage3_backend", "legacy")
    if backend == "hybrid_ab":
        from config import STAGE2_HYBRID_AB_MODE, STAGE2_HYBRID_AB_PROFILE

        return STAGE2_HYBRID_AB_PROFILE, STAGE2_HYBRID_AB_MODE, "hybrid_ab_default"
    return STAGE2_DEFAULT_PROFILE, STAGE2_DEFAULT_MODE, "global_default"


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
    # Backend-specific diagnostics payload (from Stage3EncodeResult.metrics).
    # Used by evaluation/aggregation; must not be stored back into repo_config.
    stage3_metrics: dict[str, Any] = field(default_factory=dict)

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


def stage2_funnel_metrics(stats: CleaningStats) -> dict[str, Any]:
    """Aggregate Stage2→B funnel counters (hybrid_ab telemetry).

    * ``stage2_removed_free_text_estimated_tokens`` — sum of ``ceil(inner_chars/4)``
      over **removed** low-risk docstrings that classify as B-route candidates under
      default ``SemanticClassifierConfig`` (same heuristic as Stage3 B visibility).
    * Comment token estimates use ``tokenize.COMMENT`` length / 4 (see ``lossy_cleaner``).
    """
    return {
        "stage2_removed_comment_count": int(stats.stage2_removed_comment_count),
        "stage2_removed_comment_tokens": int(stats.stage2_removed_comment_tokens_est),
        "stage2_removed_docstring_count": int(stats.stage2_removed_docstring_count),
        "stage2_removed_docstring_tokens": int(stats.stage2_removed_docstring_tokens_est),
        "stage2_removed_free_text_estimated_tokens": int(
            stats.stage2_removed_free_text_estimated_tokens
        ),
        "stage2_retained_for_b_probe_count": int(stats.stage2_retained_for_b_probe_count),
        "stage2_retained_for_b_probe_tokens": int(stats.stage2_retained_for_b_probe_tokens_est),
    }


def _stage1_vocab_intro(repo_config, tokenizer, tok_type: str) -> int:
    cands = repo_config.skeleton_candidates()
    if not cands:
        return 0
    entries = [
        build_stage1_vocab_entry(c.marker_text or make_syn_marker(i), c.skeleton)
        for i, c in enumerate(cands)
    ]
    return compute_vocab_intro_cost(
        entries,
        mode=VOCAB_COST_MODE,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )


def _stage3_encode_result(text: str, repo_config, tokenizer, tok_type: str | None):
    """Backend orchestration: encode + vocab intro payload."""
    backend = get_stage3_backend(getattr(repo_config, "stage3_backend", "legacy"))
    return backend.encode(
        text,
        repo_config,
        tokenizer=tokenizer,
        tok_type=tok_type,
    ), backend


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
    stage1_vocab_tokens = [c.marker_text or make_syn_marker(i) for i, c in enumerate(cands)]

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
    stage1_vocab_tokens = [c.marker_text or make_syn_marker(i) for i, c in enumerate(cands)]

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
    cleaned, _ = stage2_clean_skip_syn_and_stats(
        text,
        stage2_cfg.cleaning,
        mode=stage2_cfg.mode,
        drop_empty_cleaned_lines=False,
    )
    return cleaned


def apply_stage3(
    text: str,
    repo_config,
    tokenizer=None,
    tok_type: str | None = None,
) -> str:
    """Stage3 wrapper kept for backward compatibility (legacy/plan_a/hybrid_ab)."""
    res, _backend = _stage3_encode_result(text, repo_config, tokenizer, tok_type)
    return res.encoded_text


# Used by `apply_pipeline` to decide whether it can reuse the already-computed
# backend result instead of calling `apply_stage3` again (tests monkeypatch it).
_DEFAULT_APPLY_STAGE3 = apply_stage3


def apply_pipeline(
    source: str,
    repo_config,
    tokenizer,
    tok_type: str,
    count_fn=None,
    stage2_profile: str | None = None,
    stage2_mode: str | None = None,
) -> tuple[str, CompressionBreakdown]:
    if count_fn is None:
        def count_fn_local(text: str) -> int:
            return _count_with_ops(text, tokenizer, tok_type)

        count_fn = count_fn_local

    s2_profile, s2_mode, _s2_src = resolve_stage2_for_pipeline(
        repo_config, stage2_profile, stage2_mode
    )

    baseline_tokens = count_fn(source)
    after_s1 = apply_stage1_with_stats(source, repo_config, tokenizer, tok_type)[0]
    after_s1_tokens = count_fn(after_s1)

    s2_cfg = build_stage2_config(profile=s2_profile, mode=s2_mode)
    cleaning = s2_cfg.cleaning
    if getattr(repo_config, "stage3_backend", "") == "hybrid_ab":
        if os.getenv("ET_STAGE2_B_STARVATION_PROBE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            cleaning = replace(
                cleaning,
                b_starvation_probe=True,
                b_probe_min_inner_chars=int(
                    os.getenv("ET_STAGE2_B_PROBE_MIN_INNER_CHARS", "64")
                ),
                b_probe_min_space_ratio=float(
                    os.getenv("ET_STAGE2_B_PROBE_MIN_SPACE_RATIO", "0.12")
                ),
            )
    after_s2, s2_stats = stage2_clean_skip_syn_and_stats(
        after_s1,
        cleaning,
        mode=s2_cfg.mode,
        drop_empty_cleaned_lines=False,
    )
    after_s2_tokens = count_fn(after_s2)

    # Backend-driven vocab intro payload (accounting) computed once.
    stage3_result, backend = _stage3_encode_result(after_s2, repo_config, tokenizer, tok_type)
    if getattr(repo_config, "stage3_backend", "") == "hybrid_ab":
        merged = dict(stage3_result.metrics)
        merged.update(stage2_funnel_metrics(s2_stats))
        stage3_result = Stage3EncodeResult(
            encoded_text=stage3_result.encoded_text,
            vocab_entries=stage3_result.vocab_entries,
            metrics=merged,
        )
    s1_intro = _stage1_vocab_intro(repo_config, tokenizer, tok_type)
    s3_intro = backend.compute_intro_cost(
        stage3_result,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )

    # For unit tests: they often monkeypatch `apply_stage3` and expect the call.
    if apply_stage3 is _DEFAULT_APPLY_STAGE3:
        after_s3 = stage3_result.encoded_text
    else:
        try:
            after_s3 = apply_stage3(
                after_s2,
                repo_config,
                tokenizer=tokenizer,
                tok_type=tok_type,
            )
        except TypeError:
            after_s3 = apply_stage3(after_s2, repo_config)

    after_s3_tokens = count_fn(after_s3)

    breakdown = CompressionBreakdown(
        baseline_tokens=baseline_tokens,
        after_syntax=after_s1_tokens,
        after_cleaning=after_s2_tokens,
        after_replacement=after_s3_tokens,
        stage1_vocab_intro_tokens=s1_intro,
        stage2_vocab_intro_tokens=0,
        stage3_vocab_intro_tokens=s3_intro,
        stage3_metrics=stage3_result.metrics,
    )
    return after_s3, breakdown
