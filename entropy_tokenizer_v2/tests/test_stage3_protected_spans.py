from types import SimpleNamespace

from markers import get_syn_line_spans
from token_scorer import apply_token_replacement_with_protected_spans


def test_protected_syn_spans_are_not_replaced() -> None:
    text = "<SYN_0> foo bar\nfoo = bar\n"
    spans = get_syn_line_spans(text)
    out = apply_token_replacement_with_protected_spans(
        text,
        {"foo": "<VAR>", "bar": "<VAR>"},
        spans,
    )
    assert out.splitlines()[0] == "<SYN_0> foo bar"
    assert out.splitlines()[1] == "<VAR> = <VAR>"


def test_pipeline_stage3_uses_text_level_protected_spans() -> None:
    from pipeline import apply_stage3

    text = "<SYN_0> foo bar\nfoo = bar\n"
    repo_config = SimpleNamespace(replacement_map={"foo": "<VAR>", "bar": "<VAR>"})
    out = apply_stage3(text, repo_config)
    assert out.splitlines()[0] == "<SYN_0> foo bar"
    assert out.splitlines()[1] == "<VAR> = <VAR>"
