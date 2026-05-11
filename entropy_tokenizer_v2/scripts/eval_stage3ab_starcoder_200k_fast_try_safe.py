#!/usr/bin/env python3
"""
Fast-try wrapper for Stage3 hybrid_ab on the frozen 200k StarCoder baseline run.

- Re-runs `eval_stage3ab_starcoder_200k.py` in a subprocess so `config` reloads cleanly.
- Injects more aggressive A/B env overrides (see _FAST_TRY_ENV).
- Writes under `results_fast_try/stage3ab_starcoder_200k/` (ET_RESULTS_DIR).
- Uses `cache_fast_try/` (ET_CACHE_DIR) so mining does not reuse the baseline repo_config cache.

Does not modify the baseline script or overwrite `results/stage3ab_starcoder_200k/`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "eval_stage3ab_starcoder_200k.py"

# Aggressive but small-delta knobs (env-only). Tune here for the next fast try.
_FAST_TRY_ENV: dict[str, str] = {
    # A
    "ET_STAGE3_AB_A_MIN_OCC": "2",
    "ET_STAGE3_AB_MIN_RAW_TOKEN_LEN": "2",
    "ET_STAGE3_AB_MAX_ALIAS_TOKEN_LEN": "3",
    "ET_STAGE3_AB_A_ALIAS_RANK_POOL_CAP": "64",
    "ET_STAGE3_AB_A_COMBO_GREEDY": "1",
    "ET_STAGE3_AB_A_COMBO_MAX": "48",
    # B
    "ET_STAGE3_AB_B_CHANNEL_PRIORITY": "normal",
    "ET_STAGE3_AB_B_SIMILARITY_KIND": "mixed",
    "ET_STAGE3_AB_B_LEXICAL_WEIGHT": "0.6",
    "ET_STAGE3_AB_B_CHAR_WEIGHT": "0.4",
    "ET_STAGE3_AB_B_SIMILARITY_THRESHOLD": "0.78",
    "ET_STAGE3_AB_B_RISK_THRESHOLD": "0.68",
    "ET_STAGE3_AB_B_MIN_CLUSTER_SIZE": "1",
    # Stage2 funnel for B
    "ET_STAGE2_B_STARVATION_PROBE": "1",
}


def _child_environ() -> dict[str, str]:
    e = os.environ.copy()
    e["ET_RESULTS_DIR"] = str(ROOT / "results_fast_try")
    e["ET_CACHE_DIR"] = str(ROOT / "cache_fast_try")
    e.update(_FAST_TRY_ENV)
    return e


def main() -> int:
    if not TARGET.is_file():
        print(f"[fast_try_safe] missing {TARGET}", file=sys.stderr)
        return 1
    print("[fast_try_safe] ET_RESULTS_DIR ->", ROOT / "results_fast_try", flush=True)
    print("[fast_try_safe] ET_CACHE_DIR ->", ROOT / "cache_fast_try", flush=True)
    subprocess.check_call([sys.executable, str(TARGET)], cwd=str(ROOT), env=_child_environ())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
