#!/usr/bin/env python3
"""
Stage1 + Stage2 evaluation (no Stage3): adapted pipeline
``stage2_pre_safe -> stage1 -> stage2_post_surface``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AST_MIN_FREQ, VOCAB_COST_MODE, VOCAB_COST_SCOPE
from lossy_cleaner import CleaningStats, lossless_clean
from marker_count import count_augmented, encode as mc_encode
from markers import make_syn_marker
from pipeline import apply_stage1_stage2_adapted, apply_stage1_with_stats
from placeholder_accounting import compute_vocab_intro_cost
from stage2.cleaning import run_stage2_pre_safe
from stage2.config import STAGE2_ADAPTED_ORDER_LABEL, build_stage2_execution_plan
from syntax_compressor import (
    SkeletonCandidate,
    build_candidate_pool,
    greedy_mdl_select,
    mine_skeletons,
    build_stage1_vocab_entry,
)
from tokenizer_utils import GPT4oTokenizerResolutionError, resolve_gpt4o_base_tokenizer

log = logging.getLogger("eval_stage1_stage2_only")

DEFAULT_CORPUS_JSONL = "stage1_starcoder_1m_corpus.jsonl"
DEFAULT_CHECKPOINT = "stage1_starcoder_1m_checkpoint.json"
DEFAULT_MANIFEST = "stage1_starcoder_1m_manifest.json"

EXPERIMENT_BASELINE = "baseline"
EXPERIMENT_PRE_ONLY = "stage2_safe_pre_only"
EXPERIMENT_STAGE1_ONLY = "stage1_only"
EXPERIMENT_ADAPTED_SAFE = "stage1_stage2_safe_adapted"
EXPERIMENT_ADAPTED_AGG = "stage1_stage2_aggressive_upper_bound"

TEXT_PREVIEW_LEN = 420

NOTES = (
    "Adapted pipeline: stage2_pre_safe (AST docstrings + directive-preserving comment removal) "
    "-> stage1 -> stage2_post_surface (blank/ws; indent strip only for aggressive_upper_bound). "
    "No Stage3. | aggressive_upper_bound is not the default line (lossy indent)."
)


def _load_stage1_eval_module() -> Any:
    path = ROOT / "scripts" / "eval_stage1_starcoder_1m.py"
    name = "eval_stage1_starcoder_1m_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_S1: Any | None = None


def _s1() -> Any:
    global _S1
    if _S1 is None:
        _S1 = _load_stage1_eval_module()
    return _S1


def safe_parse_success(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError):
        return False
    except Exception:
        return False


def _preview(text: str, n: int = TEXT_PREVIEW_LEN) -> str:
    t = text.replace("\r\n", "\n")
    if len(t) <= n:
        return t
    return t[:n] + "\n…"


def process_one_file_experiment(
    sample: dict[str, Any],
    *,
    tokenizer: Any,
    tok_type: str,
    stage1_repo: Any | None,
    experiment_name: str,
) -> dict[str, Any]:
    path = str(sample.get("path") or "unknown.py")
    raw = sample["text"]
    clean, _ = lossless_clean(raw)
    base_tokens = int(sample.get("base_tokens", 0))

    def tok(s: str) -> int:
        return count_augmented(s, tokenizer, tok_type)

    baseline_sequence_tokens = tok(clean)
    pre_text = clean
    stage1_text = clean
    final_text = clean
    pre_stats = CleaningStats()
    post_stats = CleaningStats()
    changed_pre = False
    changed_s1 = False
    changed_post = False

    if experiment_name == EXPERIMENT_BASELINE:
        pass

    elif experiment_name == EXPERIMENT_PRE_ONLY:
        plan = build_stage2_execution_plan("safe")
        pre_text, pre_stats = run_stage2_pre_safe(clean, plan.pre_cfg, path=path)
        final_text = pre_text
        changed_pre = pre_text != clean

    elif experiment_name == EXPERIMENT_STAGE1_ONLY:
        if stage1_repo is not None:
            stage1_text, _ = apply_stage1_with_stats(
                clean, stage1_repo, tokenizer, tok_type
            )
            final_text = stage1_text
            changed_s1 = stage1_text != clean

    elif experiment_name == EXPERIMENT_ADAPTED_SAFE:
        if stage1_repo is not None:
            out = apply_stage1_stage2_adapted(
                clean,
                stage1_repo,
                stage2_profile="safe",
                tokenizer=tokenizer,
                tok_type=tok_type,
                path=path,
            )
            pre_text = out["stage2_pre_text"]
            stage1_text = out["stage1_text"]
            final_text = out["stage2_post_text"]
            pre_stats = out["stage2_pre_stats"]
            post_stats = out["stage2_post_stats"]
            changed_pre = pre_text != clean
            changed_s1 = stage1_text != pre_text
            changed_post = final_text != stage1_text

    elif experiment_name == EXPERIMENT_ADAPTED_AGG:
        if stage1_repo is not None:
            out = apply_stage1_stage2_adapted(
                clean,
                stage1_repo,
                stage2_profile="aggressive_upper_bound",
                tokenizer=tokenizer,
                tok_type=tok_type,
                path=path,
            )
            pre_text = out["stage2_pre_text"]
            stage1_text = out["stage1_text"]
            final_text = out["stage2_post_text"]
            pre_stats = out["stage2_pre_stats"]
            post_stats = out["stage2_post_stats"]
            changed_pre = pre_text != clean
            changed_s1 = stage1_text != pre_text
            changed_post = final_text != stage1_text

    else:
        raise ValueError(f"unknown experiment: {experiment_name}")

    stage2_pre_sequence_tokens = tok(pre_text)
    stage1_sequence_tokens = tok(stage1_text)
    final_sequence_tokens = tok(final_text)

    rep_pre = pre_stats.docstring_removal_report or {}
    pc = (rep_pre.get("path_context") or {}) if rep_pre else {}

    return {
        "path": path,
        "original_text_preview": _preview(clean),
        "stage2_pre_text_preview": _preview(pre_text),
        "stage1_text_preview": _preview(stage1_text),
        "final_text_preview": _preview(final_text),
        "base_tokens": base_tokens,
        "baseline_sequence_tokens": baseline_sequence_tokens,
        "stage2_pre_sequence_tokens": stage2_pre_sequence_tokens,
        "stage1_sequence_tokens": stage1_sequence_tokens,
        "final_sequence_tokens": final_sequence_tokens,
        "baseline_parse_success": safe_parse_success(clean),
        "stage2_pre_parse_success": safe_parse_success(pre_text),
        "stage1_parse_success": safe_parse_success(stage1_text),
        "final_parse_success": safe_parse_success(final_text),
        "stage2_pre_removed_blank_lines": int(pre_stats.removed_blank_lines),
        "stage2_pre_removed_docstring_chars": int(pre_stats.removed_docstring_chars),
        "stage2_pre_removed_indent_chars": int(pre_stats.removed_indent_chars),
        "stage2_pre_docstring_removed_count": int(rep_pre.get("removed_count", 0)),
        "stage2_pre_docstring_kept_count": int(rep_pre.get("kept_count", 0)),
        "stage2_pre_docstring_parse_failed": bool(rep_pre.get("parse_failed")),
        "stage2_pre_docstring_path_context_labels": ",".join(
            str(x) for x in (pc.get("matched_context_labels") or [])
        ),
        "stage2_post_removed_blank_lines": int(post_stats.removed_blank_lines),
        "stage2_post_removed_indent_chars": int(post_stats.removed_indent_chars),
        "changed_stage2_pre": changed_pre,
        "changed_stage1": changed_s1,
        "changed_stage2_post": changed_post,
        "_original_clean": clean,
        "_pre_text": pre_text,
        "_stage1_text": stage1_text,
        "_final_text": final_text,
        "_experiment_name": experiment_name,
    }


def build_stage1_mining_context(
    clean_texts: list[str],
    tokenizer: Any,
    tok_type: str,
) -> dict[str, Any]:
    s1m = _s1()
    Stage1OnlyRepoConfig = s1m.Stage1OnlyRepoConfig

    if not clean_texts:
        return {
            "repo_config": Stage1OnlyRepoConfig(selected=[]),
            "selected": [],
            "skeleton_rows": [],
            "stage1_vocab_intro_tokens": 0,
            "stage1_vocab_tokens": [],
            "baseline_sequence_tokens": 0,
            "agg_stats": {},
        }

    baseline_seq = 0
    for src in clean_texts:
        baseline_seq += count_augmented(src, tokenizer, tok_type)

    N_baseline = 0
    for src in clean_texts:
        N_baseline += len(mc_encode(tokenizer, tok_type, src))

    skeleton_counts = mine_skeletons(clean_texts, min_freq=AST_MIN_FREQ)
    candidates = build_candidate_pool(
        skeleton_counts, tokenizer, tok_type, sources=clean_texts
    )
    V0 = getattr(tokenizer, "n_vocab", None)
    if V0 is None:
        mt = getattr(tokenizer, "max_token_value", None)
        V0 = int(mt) + 1 if mt is not None else 256000
    selected: list[SkeletonCandidate] = greedy_mdl_select(candidates, N_baseline, V0)
    repo_config = Stage1OnlyRepoConfig(selected=selected)

    agg_stats: dict[str, dict[str, int]] = {}
    for src in clean_texts:
        _out, stats = apply_stage1_with_stats(
            src, repo_config, tokenizer, tok_type
        )
        for sk, st in stats.items():
            a = agg_stats.setdefault(
                sk,
                {
                    "candidate_occurrences": 0,
                    "replaced_occurrences": 0,
                    "skipped_nonpositive_occurrences": 0,
                },
            )
            a["candidate_occurrences"] += int(st.get("candidate_occurrences", 0))
            a["replaced_occurrences"] += int(st.get("replaced_occurrences", 0))
            a["skipped_nonpositive_occurrences"] += int(
                st.get("skipped_nonpositive_occurrences", 0)
            )

    entries = [
        build_stage1_vocab_entry(make_syn_marker(i), c.skeleton)
        for i, c in enumerate(selected)
    ]
    stage1_vocab_intro = compute_vocab_intro_cost(
        entries,
        mode=VOCAB_COST_MODE,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )
    vocab_tokens = [make_syn_marker(i) for i in range(len(selected))]

    skeleton_rows: list[dict[str, Any]] = []
    for i, c in enumerate(selected):
        sk = c.skeleton
        occ = agg_stats.get(sk, {})
        skeleton_rows.append(
            {
                "marker": make_syn_marker(i),
                "skeleton": sk,
                "candidate_occurrences": occ.get("candidate_occurrences", 0),
                "replaced_occurrences": occ.get("replaced_occurrences", 0),
                "skipped_nonpositive_occurrences": occ.get(
                    "skipped_nonpositive_occurrences", 0
                ),
                "total_baseline_sequence_tokens": c.total_baseline_sequence_tokens,
                "total_compressed_sequence_tokens": c.total_compressed_sequence_tokens,
                "total_sequence_net_saving": c.total_net_saving,
                "vocab_intro_tokens": c.vocab_intro_tokens,
                "effective_total_net_saving": c.effective_total_net_saving,
            }
        )

    return {
        "repo_config": repo_config,
        "selected": selected,
        "skeleton_rows": skeleton_rows,
        "stage1_vocab_intro_tokens": stage1_vocab_intro,
        "stage1_vocab_tokens": vocab_tokens,
        "baseline_sequence_tokens": baseline_seq,
        "agg_stats": agg_stats,
    }


def _amortize_vocab_intro(n_files: int, intro: int) -> list[int]:
    if n_files <= 0:
        return []
    base = intro // n_files
    rem = intro % n_files
    return [base + (1 if i < rem else 0) for i in range(n_files)]


def _aggregate_parse_rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for x in flags if x) / len(flags)


def _row_from_core(
    experiment_name: str,
    r: dict[str, Any],
    s: dict[str, Any],
    vocab_amort: int,
) -> dict[str, Any]:
    if experiment_name == EXPERIMENT_BASELINE:
        s1_eff = r["baseline_sequence_tokens"]
        fin_eff = r["final_sequence_tokens"]
    elif experiment_name == EXPERIMENT_PRE_ONLY:
        s1_eff = r["baseline_sequence_tokens"]
        fin_eff = r["final_sequence_tokens"]
    else:
        s1_eff = r["stage1_sequence_tokens"] + vocab_amort
        fin_eff = r["final_sequence_tokens"] + vocab_amort
    return {
        "experiment_name": experiment_name,
        "path": r["path"],
        "base_tokens": s.get("base_tokens", 0),
        "original_text_preview": r["original_text_preview"],
        "stage2_pre_text_preview": r["stage2_pre_text_preview"],
        "stage1_text_preview": r["stage1_text_preview"],
        "final_text_preview": r["final_text_preview"],
        "baseline_sequence_tokens": r["baseline_sequence_tokens"],
        "stage2_pre_sequence_tokens": r["stage2_pre_sequence_tokens"],
        "stage1_sequence_tokens": r["stage1_sequence_tokens"],
        "final_sequence_tokens": r["final_sequence_tokens"],
        "stage1_effective_total_tokens": s1_eff,
        "final_effective_total_tokens": fin_eff,
        "baseline_parse_success": r["baseline_parse_success"],
        "stage2_pre_parse_success": r["stage2_pre_parse_success"],
        "stage1_parse_success": r["stage1_parse_success"],
        "final_parse_success": r["final_parse_success"],
        "stage2_pre_removed_blank_lines": r["stage2_pre_removed_blank_lines"],
        "stage2_pre_removed_docstring_chars": r["stage2_pre_removed_docstring_chars"],
        "stage2_pre_removed_indent_chars": r["stage2_pre_removed_indent_chars"],
        "stage2_pre_docstring_removed_count": r["stage2_pre_docstring_removed_count"],
        "stage2_pre_docstring_kept_count": r["stage2_pre_docstring_kept_count"],
        "stage2_pre_docstring_parse_failed": r["stage2_pre_docstring_parse_failed"],
        "stage2_post_removed_blank_lines": r["stage2_post_removed_blank_lines"],
        "stage2_post_removed_indent_chars": r["stage2_post_removed_indent_chars"],
        "changed_stage2_pre": r["changed_stage2_pre"],
        "changed_stage1": r["changed_stage1"],
        "changed_stage2_post": r["changed_stage2_post"],
    }


def _summarize_rows(
    experiment_name: str,
    stage2_profile: str,
    stage2_order: str,
    rows: list[dict[str, Any]],
    ctx: dict[str, Any],
    intro: int,
) -> dict[str, Any]:
    n = len(rows)
    base_total = int(ctx["baseline_sequence_tokens"])
    t_pre = sum(x["stage2_pre_sequence_tokens"] for x in rows)
    t_s1 = sum(x["stage1_sequence_tokens"] for x in rows)
    t_fin = sum(x["final_sequence_tokens"] for x in rows)
    t_fin_eff = sum(x["final_effective_total_tokens"] for x in rows)
    t_s1_eff = sum(x["stage1_effective_total_tokens"] for x in rows)

    sum_pre_blank = sum(x["stage2_pre_removed_blank_lines"] for x in rows)
    sum_pre_doc = sum(x["stage2_pre_removed_docstring_chars"] for x in rows)
    sum_pre_ind = sum(x["stage2_pre_removed_indent_chars"] for x in rows)
    sum_pre_drm = sum(x["stage2_pre_docstring_removed_count"] for x in rows)
    sum_pre_dkp = sum(x["stage2_pre_docstring_kept_count"] for x in rows)
    doc_fail = sum(1 for x in rows if x["stage2_pre_docstring_parse_failed"])

    sum_post_blank = sum(x["stage2_post_removed_blank_lines"] for x in rows)
    sum_post_ind = sum(x["stage2_post_removed_indent_chars"] for x in rows)

    base = max(1, base_total)
    return {
        "experiment_name": experiment_name,
        "stage2_profile": stage2_profile,
        "stage2_order": stage2_order,
        "baseline_sequence_tokens": base_total,
        "baseline_effective_total_tokens": base_total,
        "baseline_parse_success_rate": _aggregate_parse_rate(
            [x["baseline_parse_success"] for x in rows]
        ),
        "stage2_pre_sequence_tokens": t_pre,
        "stage2_pre_parse_success_rate": _aggregate_parse_rate(
            [x["stage2_pre_parse_success"] for x in rows]
        ),
        "stage2_post_sequence_tokens": t_fin,
        "stage1_sequence_tokens": t_s1,
        "stage1_vocab_intro_tokens": intro,
        "stage1_effective_total_tokens": t_s1_eff,
        "stage1_sequence_reduction_ratio": 1.0 - t_s1 / base if n else 0.0,
        "stage1_effective_reduction_ratio": 1.0 - t_s1_eff / base if n else 0.0,
        "stage1_parse_success_rate": _aggregate_parse_rate(
            [x["stage1_parse_success"] for x in rows]
        ),
        "selected_skeleton_count": len(ctx["selected"]),
        "final_sequence_tokens": t_fin,
        "final_effective_total_tokens": t_fin_eff,
        "final_parse_success_rate": _aggregate_parse_rate(
            [x["final_parse_success"] for x in rows]
        ),
        "stage2_pre_total_removed_blank_lines": sum_pre_blank,
        "stage2_pre_total_removed_docstring_chars": sum_pre_doc,
        "stage2_pre_total_removed_indent_chars": sum_pre_ind,
        "stage2_pre_total_docstring_removed_count": sum_pre_drm,
        "stage2_pre_total_docstring_kept_count": sum_pre_dkp,
        "stage2_pre_docstring_parse_failed_files": doc_fail,
        "stage2_post_total_removed_blank_lines": sum_post_blank,
        "stage2_post_total_removed_indent_chars": sum_post_ind,
        "stage2_sequence_reduction_ratio": 1.0 - t_fin / max(1, t_s1) if n else 0.0,
        "stage2_effective_reduction_ratio": 1.0 - t_fin_eff / base if n else 0.0,
    }


def run_experiment_batch(
    samples: list[dict[str, Any]],
    *,
    experiment_name: str,
    tokenizer: Any,
    tok_type: str,
    ctx: dict[str, Any],
    stage2_profile: str,
    stage2_order: str,
    apply_vocab_amort: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo = ctx["repo_config"] if experiment_name != EXPERIMENT_BASELINE else None
    intro = int(ctx["stage1_vocab_intro_tokens"]) if apply_vocab_amort else 0
    amort = _amortize_vocab_intro(len(samples), intro) if apply_vocab_amort else [0] * len(samples)

    rows_out: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        r = process_one_file_experiment(
            s,
            tokenizer=tokenizer,
            tok_type=tok_type,
            stage1_repo=repo,
            experiment_name=experiment_name,
        )
        row = _row_from_core(experiment_name, r, s, amort[i] if apply_vocab_amort else 0)
        rows_out.append(row)

    summary = _summarize_rows(
        experiment_name, stage2_profile, stage2_order, rows_out, ctx, intro
    )
    return summary, rows_out


def offline_regression_samples() -> list[dict[str, Any]]:
    body = (
        "def foo():\n"
        '    """doc"""\n'
        "    return 1\n\n"
        "class Bar:\n"
        "    def _m(self):\n"
        '        """nested"""\n'
        "        pass\n"
    )
    samples: list[dict[str, Any]] = []
    for i in range(5):
        samples.append(
            {
                "path": f"tests/test_regression_{i}.py",
                "text": body * 8,
                "base_tokens": 0,
            }
        )
    return samples


def pick_adaptation_examples(
    per_file_rows: list[dict[str, Any]],
    *,
    max_examples: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (safe_examples, agg_examples) JSON-ready dicts."""
    safe_rows = [r for r in per_file_rows if r["experiment_name"] == EXPERIMENT_ADAPTED_SAFE]
    agg_rows = [r for r in per_file_rows if r["experiment_name"] == EXPERIMENT_ADAPTED_AGG]

    def pick(rows: list[dict[str, Any]], exp: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        # docstring removed
        for r in rows:
            if r["stage2_pre_docstring_removed_count"] > 0 and len(out) < max_examples:
                out.append(_example_dict(r, "docstring_removed_before_stage1", exp))
        # markerized
        for r in rows:
            if r["changed_stage1"] and "<SYN_" in r.get("stage1_text_preview", "") and len(out) < max_examples:
                if out and out[-1]["path"] == r["path"]:
                    continue
                out.append(_example_dict(r, "stage1_markerized_after_safe_preclean", exp))
        # post indent (agg only)
        for r in rows:
            if (
                exp == EXPERIMENT_ADAPTED_AGG
                and r["stage2_post_removed_indent_chars"] > 0
                and len(out) < max_examples
            ):
                out.append(_example_dict(r, "aggressive_upper_bound_indent_strip", exp))
        # noop
        for r in rows:
            if (
                not r["changed_stage2_pre"]
                and not r["changed_stage1"]
                and not r["changed_stage2_post"]
                and len(out) < max_examples
            ):
                out.append(_example_dict(r, "safe_noop_explained", exp))
        return out[:max_examples]

    return pick(safe_rows, EXPERIMENT_ADAPTED_SAFE), pick(agg_rows, EXPERIMENT_ADAPTED_AGG)


def _example_dict(r: dict[str, Any], category: str, exp: str) -> dict[str, Any]:
    seq_delta = r["stage1_sequence_tokens"] - r["stage2_pre_sequence_tokens"]
    return {
        "path": r["path"],
        "category": category,
        "experiment": exp,
        "original_excerpt": r["original_text_preview"],
        "stage2_pre_excerpt": r["stage2_pre_text_preview"],
        "stage1_excerpt": r["stage1_text_preview"],
        "final_excerpt": r["final_text_preview"],
        "stage2_pre_changes": {
            "removed_blank_lines": r["stage2_pre_removed_blank_lines"],
            "removed_docstring_chars": r["stage2_pre_removed_docstring_chars"],
            "docstring_removed_count": r["stage2_pre_docstring_removed_count"],
        },
        "stage1_changes": {
            "sequence_delta_vs_pre": seq_delta,
        },
        "stage2_post_changes": {
            "removed_blank_lines": r["stage2_post_removed_blank_lines"],
            "removed_indent_chars": r["stage2_post_removed_indent_chars"],
        },
        "notes": f"pre_parse={r['stage2_pre_parse_success']} final_parse={r['final_parse_success']}",
    }


def write_adaptation_markdown(
    examples: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Stage1 + Stage2 adaptation examples",
        "",
        "Pipeline: `stage2_pre_safe -> stage1 -> stage2_post_surface`.",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines += [
            f"## Example {i}: `{ex['category']}` ({ex.get('experiment', '')})",
            "",
            f"**path:** `{ex['path']}`",
            "",
            "### Original (after lossless baseline preview)",
            "```python",
            ex["original_excerpt"],
            "```",
            "",
            "### After Stage2 pre-safe",
            "```python",
            ex["stage2_pre_excerpt"],
            "```",
            "",
            "### After Stage1",
            "```python",
            ex["stage1_excerpt"],
            "```",
            "",
            "### Final (after Stage2 post-surface)",
            "```python",
            ex["final_excerpt"],
            "```",
            "",
            f"**Notes:** {ex.get('notes', '')}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stage1_stage2_only_reports(
    summaries: list[dict[str, Any]],
    per_file_rows: list[dict[str, Any]],
    *,
    ctx: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    tokenizer_name: str,
    resume_used: bool,
    run_status: str,
    rule_breakdown_rows: list[dict[str, Any]],
    adaptation_examples: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    common_meta = {
        "dataset_name": manifest.get("dataset_name", ""),
        "tokenizer_name": tokenizer_name,
        "token_budget": manifest.get("token_budget", 0),
        "actual_accumulated_base_tokens": manifest.get(
            "actual_accumulated_base_tokens", 0
        ),
        "num_files": manifest.get("num_files", 0),
        "token_count_mode": "augmented_sequence_tokens_placeholders_as_one",
        "vocab_cost_mode": VOCAB_COST_MODE,
        "vocab_cost_scope": VOCAB_COST_SCOPE,
        "run_status": run_status,
        "resume_used": resume_used,
        "notes": NOTES,
    }

    summary_out: list[dict[str, Any]] = []
    for s in summaries:
        row = {**common_meta, **{k: v for k, v in s.items() if not k.startswith("_")}}
        summary_out.append(row)

    (output_dir / "stage1_stage2_only_summary.json").write_text(
        json.dumps(summary_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if summary_out:
        keys = list(summary_out[0].keys())
        with (output_dir / "stage1_stage2_only_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summary_out:
                w.writerow(row)

    if per_file_rows:
        pf_keys = list(per_file_rows[0].keys())
        with (output_dir / "stage1_stage2_only_per_file.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=pf_keys)
            w.writeheader()
            for row in per_file_rows:
                w.writerow(row)

    sk_rows = ctx.get("skeleton_rows") or []
    sk_path = output_dir / "stage1_stage2_only_selected_skeletons.csv"
    if sk_rows:
        with sk_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sk_rows[0].keys()))
            w.writeheader()
            for r in sk_rows:
                w.writerow(r)
    else:
        sk_path.write_text(
            "marker,skeleton,candidate_occurrences,replaced_occurrences\n",
            encoding="utf-8",
        )

    (output_dir / "stage1_stage2_only_vocab_tokens.json").write_text(
        json.dumps({"tokens": ctx.get("stage1_vocab_tokens", [])}, indent=2),
        encoding="utf-8",
    )

    manifest_out = {
        **manifest,
        "experiments": [s["experiment_name"] for s in summaries],
        "stage2_adapted_order_default": STAGE2_ADAPTED_ORDER_LABEL,
    }
    (output_dir / "stage1_stage2_only_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if rule_breakdown_rows:
        rb_keys = list(rule_breakdown_rows[0].keys())
        with (output_dir / "stage2_rule_breakdown.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=rb_keys)
            w.writeheader()
            for row in rule_breakdown_rows:
                w.writerow(row)

    if adaptation_examples:
        (output_dir / "stage1_stage2_adaptation_examples.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in adaptation_examples)
            + "\n",
            encoding="utf-8",
        )
        write_adaptation_markdown(
            adaptation_examples,
            output_dir / "stage1_stage2_adaptation_examples.md",
        )


def _rule_breakdown_for_summary(
    profile: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage2_profile": profile,
        "stage2_pre_total_removed_blank_lines": summary.get(
            "stage2_pre_total_removed_blank_lines", 0
        ),
        "stage2_pre_total_removed_docstring_chars": summary.get(
            "stage2_pre_total_removed_docstring_chars", 0
        ),
        "stage2_pre_total_removed_indent_chars": summary.get(
            "stage2_pre_total_removed_indent_chars", 0
        ),
        "stage2_post_total_removed_blank_lines": summary.get(
            "stage2_post_total_removed_blank_lines", 0
        ),
        "stage2_post_total_removed_indent_chars": summary.get(
            "stage2_post_total_removed_indent_chars", 0
        ),
        "experiment_name": summary.get("experiment_name", ""),
    }


def run_all_experiments(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    tok_type: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    clean_texts = [lossless_clean(s["text"])[0] for s in samples]
    ctx = build_stage1_mining_context(clean_texts, tokenizer, tok_type)

    summaries: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []

    s, r = run_experiment_batch(
        samples,
        experiment_name=EXPERIMENT_BASELINE,
        tokenizer=tokenizer,
        tok_type=tok_type,
        ctx=ctx,
        stage2_profile="",
        stage2_order="(none)",
        apply_vocab_amort=False,
    )
    summaries.append(s)
    per_file.extend(r)

    s, r = run_experiment_batch(
        samples,
        experiment_name=EXPERIMENT_PRE_ONLY,
        tokenizer=tokenizer,
        tok_type=tok_type,
        ctx=ctx,
        stage2_profile="safe_pre_only",
        stage2_order="pre_safe_only",
        apply_vocab_amort=False,
    )
    summaries.append(s)
    per_file.extend(r)

    s, r = run_experiment_batch(
        samples,
        experiment_name=EXPERIMENT_STAGE1_ONLY,
        tokenizer=tokenizer,
        tok_type=tok_type,
        ctx=ctx,
        stage2_profile="",
        stage2_order="(none)",
        apply_vocab_amort=True,
    )
    summaries.append(s)
    per_file.extend(r)

    s, r = run_experiment_batch(
        samples,
        experiment_name=EXPERIMENT_ADAPTED_SAFE,
        tokenizer=tokenizer,
        tok_type=tok_type,
        ctx=ctx,
        stage2_profile="safe",
        stage2_order=STAGE2_ADAPTED_ORDER_LABEL,
        apply_vocab_amort=True,
    )
    summaries.append(s)
    per_file.extend(r)
    rule_rows.append(_rule_breakdown_for_summary("safe_adapted", s))

    s, r = run_experiment_batch(
        samples,
        experiment_name=EXPERIMENT_ADAPTED_AGG,
        tokenizer=tokenizer,
        tok_type=tok_type,
        ctx=ctx,
        stage2_profile="aggressive_upper_bound",
        stage2_order=STAGE2_ADAPTED_ORDER_LABEL,
        apply_vocab_amort=True,
    )
    summaries.append(s)
    per_file.extend(r)
    rule_rows.append(_rule_breakdown_for_summary("aggressive_upper_bound_adapted", s))

    ex_safe, ex_agg = pick_adaptation_examples(per_file)
    adaptation_examples = ex_safe + ex_agg

    return summaries, per_file, ctx, rule_rows, adaptation_examples


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Stage1+Stage2 adapted evaluation (no Stage3)."
    )
    parser.add_argument("--token-budget", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-file-base-tokens", type=int, default=200_000)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="Offline smoke (alias: --mode offline).",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "offline"),
        default="full",
        help="offline: small fixed samples; full: corpus collect or skip-collect.",
    )
    args = parser.parse_args()
    if args.mode == "offline":
        args.regression_only = True

    try:
        tokenizer = resolve_gpt4o_base_tokenizer()
    except GPT4oTokenizerResolutionError as e:
        log.error("%s", e)
        return 1

    tok_type = "tiktoken"
    tokenizer_name = "gpt-4o/o200k_base (tiktoken)"

    if args.regression_only:
        samples = offline_regression_samples()
        manifest = {
            "dataset_name": "offline_regression",
            "token_budget": 0,
            "actual_accumulated_base_tokens": 0,
            "num_files": len(samples),
            "run_status": "regression_only",
        }
        run_status = "regression_only"
        resume_used = False
    else:
        corpus_jsonl = args.cache_dir / DEFAULT_CORPUS_JSONL
        checkpoint_path = args.cache_dir / DEFAULT_CHECKPOINT
        manifest_path = args.cache_dir / DEFAULT_MANIFEST
        resume_used = bool(args.resume)
        s1m = _s1()
        if args.skip_collect:
            if not corpus_jsonl.exists():
                log.error("No corpus at %s", corpus_jsonl)
                return 1
            samples_text: list[str] = []
            total_tok = 0
            with corpus_jsonl.open(encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    samples_text.append(o["text"])
                    total_tok += int(o.get("base_tokens", 0))
            samples = [
                {
                    "path": f"corpus/sample_{i}.py",
                    "text": t,
                    "base_tokens": 0,
                }
                for i, t in enumerate(samples_text)
            ]
            manifest = {
                "dataset_name": "from_cache_only",
                "token_budget": args.token_budget,
                "actual_accumulated_base_tokens": total_tok,
                "num_files": len(samples),
                "corpus_cache_path": str(corpus_jsonl),
                "run_status": "skip_collect",
            }
            run_status = "skip_collect"
        else:
            try:
                texts, manifest = s1m.collect_starcoder_python_corpus(
                    args.token_budget,
                    checkpoint_path=checkpoint_path,
                    manifest_path=manifest_path,
                    corpus_jsonl_path=corpus_jsonl,
                    resume=args.resume,
                    max_file_base_tokens=args.max_file_base_tokens,
                    dataset_override=args.dataset,
                    language=args.language,
                )
            except Exception:
                log.exception("Corpus collection failed")
                args.output_dir.mkdir(parents=True, exist_ok=True)
                err_manifest = {"run_status": "collect_failed"}
                (args.output_dir / "stage1_stage2_only_manifest.json").write_text(
                    json.dumps(err_manifest, indent=2),
                    encoding="utf-8",
                )
                return 1
            samples = [
                {
                    "path": f"corpus/sample_{i}.py",
                    "text": t,
                    "base_tokens": 0,
                }
                for i, t in enumerate(texts)
            ]
            run_status = str(manifest.get("run_status", "unknown"))

    log.info("Running adapted Stage1+Stage2 on %d files …", len(samples))
    summaries, per_file, ctx, rule_rows, adaptation_examples = run_all_experiments(
        samples, tokenizer=tokenizer, tok_type=tok_type
    )

    manifest.setdefault("num_files", len(samples))
    write_stage1_stage2_only_reports(
        summaries,
        per_file,
        ctx=ctx,
        manifest=manifest,
        output_dir=args.output_dir,
        tokenizer_name=tokenizer_name,
        resume_used=resume_used,
        run_status=run_status,
        rule_breakdown_rows=rule_rows,
        adaptation_examples=adaptation_examples,
    )
    log.info("Wrote reports under %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
