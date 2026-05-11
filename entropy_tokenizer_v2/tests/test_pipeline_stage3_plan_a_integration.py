"""Pipeline Stage3 legacy vs Plan A smoke."""

from __future__ import annotations

from types import SimpleNamespace

from pipeline import apply_pipeline, apply_stage3
from repo_miner import mine_from_sources
from config import EVAL_TOKENIZERS


def _tiny_plan_a_config() -> RepoConfig:
    return mine_from_sources(
        ["alpha = 1\nbeta = alpha\n", "gamma = 'x'\n"],
        tokenizer_key="gpt4",
        tokenizer_cfg=EVAL_TOKENIZERS["gpt4"],
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend="plan_a",
    )


def test_apply_stage3_legacy_unchanged():
    text = "foo = 1\n"
    rc = SimpleNamespace(
        stage3_backend="legacy",
        replacement_map={"foo": "<VAR>"},
    )
    out = apply_stage3(text, rc)
    assert "<VAR>" in out


def test_apply_stage3_plan_a_runs():
    rc = _tiny_plan_a_config()
    text = "alpha = 1\n"
    out = apply_stage3(text, rc)
    assert isinstance(out, str)


def test_full_pipeline_plan_a_no_crash():
    rc = _tiny_plan_a_config()
    from repo_miner import _load_tokenizer

    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    src = "alpha = 1\nbeta = 2\n"
    compressed, bd = apply_pipeline(src, rc, tok, tt)
    assert bd.baseline_tokens >= 1
    assert isinstance(compressed, str)


def test_full_pipeline_legacy_no_crash():
    rc = mine_from_sources(
        ["foo = 1\n"],
        tokenizer_key="gpt4",
        tokenizer_cfg=EVAL_TOKENIZERS["gpt4"],
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend="legacy",
    )
    from repo_miner import _load_tokenizer

    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    compressed, bd = apply_pipeline("foo = 1\n", rc, tok, tt)
    assert isinstance(compressed, str)
