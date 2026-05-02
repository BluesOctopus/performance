from types import SimpleNamespace

from pipeline import apply_stage3


def test_stage3_skip_syn_lines() -> None:
    text = "<SYN_0> foo bar\nfoo = bar"
    repo_config = SimpleNamespace(replacement_map={"foo": "<VAR>", "bar": "<ATTR>"})
    out = apply_stage3(text, repo_config)
    assert out.splitlines()[0] == "<SYN_0> foo bar"
    assert out.splitlines()[1] == "<VAR> = bar"
