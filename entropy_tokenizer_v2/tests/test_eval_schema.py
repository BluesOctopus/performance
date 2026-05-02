from __future__ import annotations

from eval.v2_eval import EvalResult


def test_eval_schema_has_unified_stage3_fields() -> None:
    fields = {f.name for f in EvalResult.__dataclass_fields__.values()}
    for name in (
        "stage3_selected_units",
        "stage3_selected_units_exact",
        "stage3_selected_units_semantic",
        "stage3_used_units_exact",
        "stage3_used_units_semantic",
        "stage3_vocab_scope",
        "stage3_vocab_scope_detail",
        "stage3_ab_mode",
        "stage3_ab_similarity_kind",
    ):
        assert name in fields
    assert "stage3_assignments" not in fields

