from __future__ import annotations

from validate.compare_answers import compare_smoke_payload
from validate.llm_smoke import run_llm_smoke
from validate.question_gen import generate_questions


def test_question_gen_has_three_types():
    qs = generate_questions("x = 1\nprint(x)\n")
    assert len(qs) == 3
    assert {q.qtype for q in qs} == {"code_qa", "variable_trace", "bug_hint"}


def test_llm_smoke_json_shape_and_compare():
    payload = run_llm_smoke(
        [
            ("a = 1\nprint(a)\n", "a = 1\nprint(a)\n"),
            ("user_id = req.get('id')\n", "uid = req.get('id')\n"),
        ],
        max_samples=2,
    )
    assert payload["n_samples"] == 2
    assert payload["questions_per_sample"] == 3
    s = compare_smoke_payload(payload)
    assert 0.0 <= s.qa_overlap <= 1.0
    assert 0.0 <= s.trace_f1 <= 1.0
    assert 0.0 <= s.bug_keyword_overlap <= 1.0

