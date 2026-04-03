from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EVAL = _ROOT / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from v2_eval import EvalResult


def test_eval_result_no_duplicate_sequence_fields():
    fields = {f.name for f in EvalResult.__dataclass_fields__.values()}
    assert "final_tokens" not in fields
    assert "reduction_pct" not in fields
    assert "sequence_final_tokens" in fields
    assert "sequence_reduction_pct" in fields


def test_eval_result_has_unified_stage3_unit_fields():
    fields = {f.name for f in EvalResult.__dataclass_fields__.values()}
    for name in (
        "stage3_selected_units",
        "stage3_selected_units_exact",
        "stage3_selected_units_semantic",
        "stage3_used_units_exact",
        "stage3_used_units_semantic",
    ):
        assert name in fields
    assert "stage3_assignments" not in fields
