from __future__ import annotations

from types import SimpleNamespace

from stage3.backends import Stage3EncodeResult, get_stage3_backend


def test_stage3_backend_interface_legacy_smoke() -> None:
    backend = get_stage3_backend("legacy")
    repo_config = SimpleNamespace(stage3_backend="legacy", replacement_map={"foo": "<VAR>"})

    res = backend.encode("foo = 1\n", repo_config, tokenizer=None, tok_type=None)
    assert isinstance(res, Stage3EncodeResult)
    assert isinstance(res.encoded_text, str)
    assert isinstance(res.vocab_entries, list)

    intro_cost = backend.compute_intro_cost(res, tokenizer=None, tok_type=None)
    assert isinstance(intro_cost, int)


def test_stage3_backend_interface_plan_a_smoke_empty_books() -> None:
    backend = get_stage3_backend("plan_a")
    repo_config = SimpleNamespace(
        stage3_backend="plan_a",
        stage3_plan_a_codebooks={},
        stage3_escape_prefix="__L__",
    )

    res = backend.encode("alpha = 1\n", repo_config, tokenizer=None, tok_type=None)
    assert isinstance(res, Stage3EncodeResult)
    assert res.encoded_text == "alpha = 1\n"
    assert res.vocab_entries == []
    assert res.meta == {}

    intro_cost = backend.compute_intro_cost(res, tokenizer=None, tok_type=None)
    assert intro_cost == 0


def test_stage3_backend_interface_hybrid_ab_smoke_missing_cfg() -> None:
    backend = get_stage3_backend("hybrid_ab")
    repo_config = SimpleNamespace(stage3_backend="hybrid_ab", stage3_ab_summary={})

    # Provide dummy tokenizer/tok_type so we reach the "missing cfg" branch.
    res = backend.encode("x = 1\n", repo_config, tokenizer=object(), tok_type="hf")
    assert isinstance(res, Stage3EncodeResult)
    assert res.encoded_text == "x = 1\n"
    assert res.vocab_entries == []
    # Missing cfg should produce a runtime warning meta (for diagnostics), not an exception.
    assert "stage3_ab_runtime_warning" in res.meta

    intro_cost = backend.compute_intro_cost(res, tokenizer=None, tok_type=None)
    assert intro_cost == 0

