from __future__ import annotations

from eval.v2_eval import EvalResult


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
    assert "hybrid_ab_stage1_override_used" in fields
    assert "hybrid_ab_stage2_override_used" in fields
    assert "stage2_resolution_source" in fields
    assert "stage3_ab_telemetry_sum_stage2_seq" in fields
    assert "stage3_ab_telemetry_sum_stage3_seq" in fields
    assert "stage3_ab_telemetry_guardrail_triggered_nfiles" in fields
    assert "stage2_removed_comment_count_sum" in fields
    assert "b_free_text_candidates_visible_after_stage2_sum" in fields
    assert "stage3_ab_after_a_tokens_sum" in fields
