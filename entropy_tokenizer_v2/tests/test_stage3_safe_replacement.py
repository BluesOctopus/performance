from token_scorer import apply_token_replacement


def test_identifier_replacement_does_not_mutate_string_contents() -> None:
    text = 'print("hello world")'
    out = apply_token_replacement(text, {"hello": "<VAR>"})
    assert out == text


def test_string_literal_replacement_matches_whole_token() -> None:
    text = 'x = "abc"\ny = "abcx"'
    out = apply_token_replacement(text, {'"abc"': "<STR>"})
    assert out.splitlines()[0] == "x = <STR>"
    assert out.splitlines()[1] == 'y = "abcx"'


def test_name_replacement_matches_identifier_token() -> None:
    text = "name = 1\nother_name = name + 2"
    out = apply_token_replacement(text, {"name": "<VAR>"})
    assert out.splitlines()[0] == "<VAR> = 1"
    assert out.splitlines()[1] == "other_name = <VAR> + 2"


def test_number_literal_replacement_only_when_whole_token_matches() -> None:
    text = "x = 12\ny = 123"
    out = apply_token_replacement(text, {"12": "<NUM>"})
    assert out.splitlines()[0] == "x = <NUM>"
    assert out.splitlines()[1] == "y = 123"
