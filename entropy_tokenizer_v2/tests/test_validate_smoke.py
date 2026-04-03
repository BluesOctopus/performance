from __future__ import annotations

import pytest

import validate.run_smoke as run_smoke_mod
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


_TINY_PY = '\ndef foo():\n    x = 1\n    y = "hello world example"\n    return x + len(y)\n'


def test_artifact_filename_for_ab_mode():
    assert (
        run_smoke_mod.artifact_filename_for_ab_mode("exact_only")
        == "validate_smoke_hybrid_ab_exact_only.json"
    )
    assert (
        run_smoke_mod.artifact_filename_for_ab_mode("hybrid")
        == "validate_smoke_hybrid_ab_hybrid.json"
    )


def test_run_validate_smoke_hybrid_ab_exact_only(monkeypatch):
    monkeypatch.delenv("ET_STAGE3_BACKEND", raising=False)
    tiny = [_TINY_PY] * 8
    monkeypatch.setattr(run_smoke_mod, "collect_py_sources", lambda _root: tiny)
    out = run_smoke_mod.run_validate_smoke(
        max_samples=5,
        stage3_ab_mode="exact_only",
        enable_b=False,
    )
    assert out["meta"]["stage3_backend"] == "hybrid_ab"
    assert out["meta"]["stage3_ab_mode"] == "exact_only"
    assert out["meta"]["enable_b"] is False
    assert "payload" in out and "summary" in out
    s = out["summary"]
    for k in (
        "qa_overlap",
        "raw_tokens_total",
        "compressed_tokens_total",
        "token_delta",
        "token_reduction_pct",
    ):
        assert k in s
    assert out["raw_tokens_total"] == s["raw_tokens_total"]
    assert out["token_delta"] == s["token_delta"]


def test_run_validate_smoke_hybrid_ab_hybrid(monkeypatch):
    monkeypatch.delenv("ET_STAGE3_BACKEND", raising=False)
    tiny = [_TINY_PY] * 8
    monkeypatch.setattr(run_smoke_mod, "collect_py_sources", lambda _root: tiny)
    out = run_smoke_mod.run_validate_smoke(
        max_samples=5,
        stage3_ab_mode="hybrid",
        enable_b=True,
    )
    assert out["meta"]["stage3_backend"] == "hybrid_ab"
    assert out["meta"]["stage3_ab_mode"] == "hybrid"
    assert out["meta"]["enable_b"] is True


def test_validate_smoke_rejects_non_hybrid_backend_env(monkeypatch):
    monkeypatch.setenv("ET_STAGE3_BACKEND", "legacy")
    tiny = [_TINY_PY] * 4
    monkeypatch.setattr(run_smoke_mod, "collect_py_sources", lambda _root: tiny)
    with pytest.raises(ValueError, match="hybrid_ab"):
        run_smoke_mod.run_validate_smoke(max_samples=2, stage3_ab_mode="exact_only")

