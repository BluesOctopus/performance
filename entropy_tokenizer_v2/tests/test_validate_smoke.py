from __future__ import annotations

from validate.question_gen import generate_questions_for_sample
from validate.llm_smoke import SmokeSample, run_validation_smoke
from validate.compare_answers import compare_smoke_answers


def test_question_gen_three_types():
    qs = generate_questions_for_sample("s1")
    assert len(qs) == 3
    assert {q.qtype for q in qs} == {"code_qa", "variable_tracing", "bug_hint"}


def test_validate_smoke_and_compare_runs():
    samples = [
        SmokeSample(
            sample_id="s1",
            original_text="def add(a,b): return a+b",
            compressed_text="def add(a,b): return a+b",
        )
    ]
    payload = run_validation_smoke(samples)
    comp = compare_smoke_answers(payload)
    assert payload["n_samples"] == 1
    assert "overall_overlap" in comp
    assert "rollback_suggested" in comp
