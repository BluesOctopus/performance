"""Minimal hybrid_ab Stage1/Stage2 ablation (80 HF samples). Writes results/stage3_hybrid_ab_s12_ablation.csv.

Token columns:
- sum_* / stage1_vocab_intro_tokens / sum_effective_total_tokens: same counters as v2_eval /
  apply_pipeline (placeholder-aware sequence counting).
- telemetry_*: summed from hybrid_ab per-file meta (guardrail uses raw tokenizer encode length);
  may differ slightly from sum_after_stage2/3 when Stage1 syntax markers are present.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from eval import bootstrap_v2  # noqa: E402

bootstrap_v2.ensure()

from config import EVAL_TOKENIZERS, STAGE2_DEFAULT_MODE, STAGE2_DEFAULT_PROFILE  # noqa: E402
from eval.v2_eval import evaluate, eval_mining_cache_name, load_eval_samples  # noqa: E402
from repo_miner import mine_from_sources  # noqa: E402

COLS = [
    "label",
    "tokenizer_key",
    "stage3_ab_mode",
    "stage2_resolution_source",
    "stage2_profile",
    "stage2_mode",
    "hybrid_ab_stage1_override_used",
    "hybrid_ab_stage2_override_used",
    "k_star_syntax",
    # Corpus-wide sequence token totals (sum over files; same counter as apply_pipeline).
    "sum_baseline_tokens",
    "sum_after_syntax_tokens",
    "sum_after_stage2_tokens",
    "sum_after_stage3_tokens",
    "sum_replacement_sequence_saved",
    # Vocab intro (Stage1 corpus-once + Stage3 request-local sum in hybrid_ab accounting).
    "stage1_vocab_intro_tokens",
    "stage3_vocab_intro_tokens",
    "sum_effective_total_tokens",
    # hybrid_ab A/B channel bookkeeping (summed local estimates; B often ~0 net).
    "sum_stage3_ab_a_sequence_saved",
    "sum_stage3_ab_b_sequence_saved",
    "sum_stage3_ab_a_intro_tokens",
    "sum_stage3_ab_b_intro_tokens",
    # Guardrail telemetry summed over files (stage2/3 tokens from meta should match aggregates above).
    "telemetry_sum_stage2_seq",
    "telemetry_sum_stage3_seq",
    "telemetry_sum_realized_delta",
    "telemetry_guardrail_triggered_nfiles",
    "telemetry_sum_aliases_rolled_back",
    "sequence_reduction_pct",
    "effective_total_reduction_pct",
    "syntax_pct",
    "cleaning_pct",
    "replacement_pct",
]


def _ab_env(mode: str) -> None:
    os.environ["ET_STAGE3_AB_MODE"] = mode
    os.environ["ET_STAGE3_AB_ENABLE_B"] = "1" if mode == "hybrid" else "0"


def main() -> None:
    n = int(os.environ.get("ET_STAGE3_COMPARE_NUM_SAMPLES", "80"))
    samples = load_eval_samples(n)
    rows: list[dict] = []

    scenarios = [
        (
            "gpt4_hybrid_ab_exact_only_legacy_s12",
            "gpt4",
            "exact_only",
            STAGE2_DEFAULT_PROFILE,
            STAGE2_DEFAULT_MODE,
        ),
        (
            "gpt4_hybrid_ab_exact_only_aggressive_s12",
            "gpt4",
            "exact_only",
            None,
            None,
        ),
        (
            "gpt4_hybrid_ab_hybrid_aggressive_s12",
            "gpt4",
            "hybrid",
            None,
            None,
        ),
        (
            "gpt2_hybrid_ab_exact_only_aggressive_s12",
            "gpt2",
            "exact_only",
            None,
            None,
        ),
    ]

    for label, tok_key, ab_mode, s2p, s2m in scenarios:
        _ab_env(ab_mode)
        cfg = EVAL_TOKENIZERS[tok_key]
        rc = mine_from_sources(
            sources=samples,
            tokenizer_key=tok_key,
            tokenizer_cfg=cfg,
            cache_name=eval_mining_cache_name(len(samples), "hybrid_ab", s2p, s2m),
            cache=True,
            verbose=True,
            stage3_backend="hybrid_ab",
        )
        r = evaluate(samples, rc, tok_key, cfg, stage2_profile=s2p, stage2_mode=s2m)
        rows.append(
            {
                "label": label,
                "tokenizer_key": r.tokenizer_key,
                "stage3_ab_mode": r.stage3_ab_mode,
                "stage2_resolution_source": r.stage2_resolution_source,
                "stage2_profile": r.stage2_profile,
                "stage2_mode": r.stage2_mode,
                "hybrid_ab_stage1_override_used": r.hybrid_ab_stage1_override_used,
                "hybrid_ab_stage2_override_used": r.hybrid_ab_stage2_override_used,
                "k_star_syntax": r.k_star_syntax,
                "sum_baseline_tokens": r.baseline_tokens,
                "sum_after_syntax_tokens": r.syntax_tokens,
                "sum_after_stage2_tokens": r.cleaning_tokens,
                "sum_after_stage3_tokens": r.sequence_final_tokens,
                "sum_replacement_sequence_saved": r.replacement_saved,
                "stage1_vocab_intro_tokens": r.stage1_vocab_intro_tokens,
                "stage3_vocab_intro_tokens": r.stage3_vocab_intro_tokens,
                "sum_effective_total_tokens": r.effective_total_tokens,
                "sum_stage3_ab_a_sequence_saved": r.stage3_ab_a_sequence_saved,
                "sum_stage3_ab_b_sequence_saved": r.stage3_ab_b_sequence_saved,
                "sum_stage3_ab_a_intro_tokens": r.stage3_ab_a_intro_tokens,
                "sum_stage3_ab_b_intro_tokens": r.stage3_ab_b_intro_tokens,
                "telemetry_sum_stage2_seq": r.stage3_ab_telemetry_sum_stage2_seq,
                "telemetry_sum_stage3_seq": r.stage3_ab_telemetry_sum_stage3_seq,
                "telemetry_sum_realized_delta": r.stage3_ab_telemetry_sum_realized_delta,
                "telemetry_guardrail_triggered_nfiles": r.stage3_ab_telemetry_guardrail_triggered_nfiles,
                "telemetry_sum_aliases_rolled_back": r.stage3_ab_telemetry_sum_aliases_rolled_back,
                "sequence_reduction_pct": f"{r.sequence_reduction_pct:.6f}",
                "effective_total_reduction_pct": f"{r.effective_total_reduction_pct:.6f}",
                "syntax_pct": f"{r.syntax_pct:.6f}",
                "cleaning_pct": f"{r.cleaning_pct:.6f}",
                "replacement_pct": f"{r.replacement_pct:.6f}",
            }
        )
        print(
            f"[ablation] {label}: Seq%={r.sequence_reduction_pct:.2f} "
            f"Eff%={r.effective_total_reduction_pct:.2f} "
            f"S2={r.stage2_profile}/{r.stage2_mode} src={r.stage2_resolution_source} "
            f"K*={r.k_star_syntax}"
        )

    out = ROOT / "results" / "stage3_hybrid_ab_s12_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[run_hybrid_ab_s12_ablation] Wrote {out}")


if __name__ == "__main__":
    main()
