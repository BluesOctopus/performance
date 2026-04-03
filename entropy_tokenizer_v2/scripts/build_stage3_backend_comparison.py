"""Regenerate results/stage3_backend_comparison.csv (HF eval corpus, same shape as before)."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from eval import bootstrap_v2  # noqa: E402

bootstrap_v2.ensure()

from config import STAGE2_DEFAULT_MODE, STAGE2_DEFAULT_PROFILE  # noqa: E402
from eval.v2_eval import EvalResult, evaluate, load_eval_samples  # noqa: E402
from repo_miner import mine_from_sources  # noqa: E402
from config import EVAL_TOKENIZERS  # noqa: E402


CSV_COLUMNS = [
    "tokenizer_key",
    "stage2_profile",
    "stage2_mode",
    "stage3_backend",
    "stage3_plan_a_profile",
    "stage3_ab_mode",
    "n_files",
    "baseline_tokens",
    "sequence_final_tokens",
    "sequence_reduction_pct",
    "effective_total_tokens",
    "effective_total_reduction_pct",
    "stage1_vocab_intro_tokens",
    "stage3_vocab_intro_tokens",
    "final_vocab_intro_tokens",
    "stage3_component_saved",
    "stage3_selected_units",
    "stage3_selected_units_exact",
    "stage3_selected_units_semantic",
    "stage3_used_units_exact",
    "stage3_used_units_semantic",
    "stage3_vocab_scope",
    "stage3_vocab_scope_detail",
    "notes",
]


def _clear_ab_env() -> None:
    for k in (
        "ET_STAGE3_AB_MODE",
        "ET_STAGE3_AB_ENABLE_B",
        "ET_STAGE3_AB_ENABLE_MID_FREE_TEXT",
        "ET_STAGE3_AB_ALLOW_MULTILINE_WHITELIST",
        "ET_STAGE3_AB_B_SIMILARITY_KIND",
    ):
        os.environ.pop(k, None)


def _run_backend(
    *,
    backend: str,
    tokenizer_keys: list[str],
    samples: list[str],
    stage2_profile: str,
    stage2_mode: str,
    env_patch: dict[str, str] | None = None,
    clear_plan_a_profile: bool = False,
) -> list[EvalResult]:
    _clear_ab_env()
    if clear_plan_a_profile:
        os.environ.pop("ET_STAGE3_PLAN_A_PROFILE", None)
    if env_patch:
        os.environ.update(env_patch)
    results: list[EvalResult] = []
    for tok_key in tokenizer_keys:
        cfg = EVAL_TOKENIZERS[tok_key]
        rc = mine_from_sources(
            sources=samples,
            tokenizer_key=tok_key,
            tokenizer_cfg=cfg,
            cache_name=f"eval_{len(samples)}_{stage2_profile}_{stage2_mode}",
            cache=True,
            verbose=True,
            stage3_backend=backend,
        )
        r = evaluate(
            samples,
            rc,
            tok_key,
            cfg,
            stage2_profile=stage2_profile,
            stage2_mode=stage2_mode,
        )
        results.append(r)
    _clear_ab_env()
    if clear_plan_a_profile:
        os.environ.pop("ET_STAGE3_PLAN_A_PROFILE", None)
    return results


def _row_from_result(r: EvalResult, notes: str) -> dict[str, str | int | float]:
    return {
        "tokenizer_key": r.tokenizer_key,
        "stage2_profile": r.stage2_profile,
        "stage2_mode": r.stage2_mode,
        "stage3_backend": r.stage3_backend,
        "stage3_plan_a_profile": r.stage3_plan_a_profile or "",
        "stage3_ab_mode": r.stage3_ab_mode or "",
        "n_files": r.n_files,
        "baseline_tokens": r.baseline_tokens,
        "sequence_final_tokens": r.sequence_final_tokens,
        "sequence_reduction_pct": r.sequence_reduction_pct,
        "effective_total_tokens": r.effective_total_tokens,
        "effective_total_reduction_pct": r.effective_total_reduction_pct,
        "stage1_vocab_intro_tokens": r.stage1_vocab_intro_tokens,
        "stage3_vocab_intro_tokens": r.stage3_vocab_intro_tokens,
        "final_vocab_intro_tokens": r.final_vocab_intro_tokens,
        "stage3_component_saved": r.stage3_component_saved,
        "stage3_selected_units": r.stage3_selected_units,
        "stage3_selected_units_exact": r.stage3_selected_units_exact,
        "stage3_selected_units_semantic": r.stage3_selected_units_semantic,
        "stage3_used_units_exact": r.stage3_used_units_exact,
        "stage3_used_units_semantic": r.stage3_used_units_semantic,
        "stage3_vocab_scope": r.stage3_vocab_scope,
        "stage3_vocab_scope_detail": r.stage3_vocab_scope_detail,
        "notes": notes,
    }


def main() -> None:
    profile = os.environ.get("ET_STAGE2_PROFILE", STAGE2_DEFAULT_PROFILE)
    mode = os.environ.get("ET_STAGE2_MODE", STAGE2_DEFAULT_MODE)
    # Default 80 matches the versioned results/stage3_backend_comparison.csv in-repo.
    n = int(os.environ.get("ET_STAGE3_COMPARE_NUM_SAMPLES", "80"))

    samples = load_eval_samples(n)
    rows: list[dict[str, str | int | float]] = []

    # 1–2 legacy (gpt4, gpt2)
    for r in _run_backend(
        backend="legacy",
        tokenizer_keys=["gpt4", "gpt2"],
        samples=samples,
        stage2_profile=profile,
        stage2_mode=mode,
        clear_plan_a_profile=True,
    ):
        tag = "s3legacy_gpt4" if r.tokenizer_key == "gpt4" else "s3legacy_gpt2"
        rows.append(_row_from_result(r, f"from v2_eval_detail_{tag}.json (regen)"))

    # 3–4 plan_a
    for r in _run_backend(
        backend="plan_a",
        tokenizer_keys=["gpt4", "gpt2"],
        samples=samples,
        stage2_profile=profile,
        stage2_mode=mode,
        clear_plan_a_profile=True,
    ):
        tag = (
            "stage3_plan_a_gpt4_conservative"
            if r.tokenizer_key == "gpt4"
            else "gpt2_plan_a"
        )
        rows.append(_row_from_result(r, f"from v2_eval_detail_{tag}.json (regen)"))

    # 5–6 hybrid_ab exact_only
    for r in _run_backend(
        backend="hybrid_ab",
        tokenizer_keys=["gpt4", "gpt2"],
        samples=samples,
        stage2_profile=profile,
        stage2_mode=mode,
        env_patch={
            "ET_STAGE3_AB_MODE": "exact_only",
            "ET_STAGE3_AB_ENABLE_B": "0",
        },
    ):
        tag = (
            "stage3_hybrid_ab_gpt4_exact_only"
            if r.tokenizer_key == "gpt4"
            else "stage3_hybrid_ab_gpt2_exact_only"
        )
        rows.append(_row_from_result(r, f"from v2_eval_detail_{tag}.json (regen)"))

    # 7 hybrid_ab hybrid (gpt4 only, matches historical CSV)
    for r in _run_backend(
        backend="hybrid_ab",
        tokenizer_keys=["gpt4"],
        samples=samples,
        stage2_profile=profile,
        stage2_mode=mode,
        env_patch={
            "ET_STAGE3_AB_MODE": "hybrid",
            "ET_STAGE3_AB_ENABLE_B": "1",
        },
    ):
        rows.append(
            _row_from_result(
                r,
                "optional B-on run from v2_eval_detail_stage3_hybrid_ab_gpt4_hybrid.json (regen)",
            )
        )

    out = ROOT / "results" / "stage3_backend_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n[build_stage3_backend_comparison] Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
