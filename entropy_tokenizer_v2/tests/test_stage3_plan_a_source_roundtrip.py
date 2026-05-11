"""Roundtrip tests for Plan A source codec (token boundaries + SYN protection)."""

from __future__ import annotations

from stage3.literal_codec.types import CodeAssignment, FieldCodebook
from stage3.literal_codec.pipeline.source_codec import (
    decode_python_source_plan_a,
    encode_python_source_plan_a,
    verify_roundtrip_plan_a,
)


def _book(
    field: str,
    pairs: list[tuple[str, str]],
    escape: str = "__L__",
) -> FieldCodebook:
    asg = [
        CodeAssignment(literal=l, code=c, raw_token_cost=3, code_token_cost=1, expected_gain=0.1)
        for l, c in pairs
    ]
    return FieldCodebook(
        field_name=field,
        version="t",
        assignments=asg,
        escape_prefix=escape,
        metadata={},
    )


def test_name_variable_and_attribute_roundtrip():
    books = {
        "variable": _book("variable", [("foo", "a"), ("bar", "b")]),
        "attribute": _book("attribute", [("baz", "c")]),
    }
    src = "foo = 1\nx = obj.baz\nbar = 2\n"
    assert verify_roundtrip_plan_a(src, books, escape_prefix="__L__")
    enc = encode_python_source_plan_a(src, books, escape_prefix="__L__")
    assert "__L__" in enc


def test_string_roundtrip_preserves_quotes_via_lookup():
    books = {
        "string": _book("string", [("'hello'", "s1"), ('"world"', "s2")]),
    }
    src = "x = 'hello'\ny = \"world\"\n"
    assert verify_roundtrip_plan_a(src, books, escape_prefix="__L__")


def test_syn_line_not_compressed():
    books = {
        "variable": _book("variable", [("foo", "a")]),
    }
    src = "<SYN_0> foo\nfoo = 1\n"
    enc = encode_python_source_plan_a(src, books, escape_prefix="__L__")
    assert enc.splitlines()[0] == "<SYN_0> foo"
    assert verify_roundtrip_plan_a(src, books, escape_prefix="__L__")


def test_double_escape_name_starts_with_escape_prefix():
    books: dict[str, FieldCodebook] = {"variable": _book("variable", [])}
    src = "__L__raw = 1\n"
    assert verify_roundtrip_plan_a(src, books, escape_prefix="__L__")


def test_multiline_string_skipped_unchanged():
    books = {
        "string": _book("string", [("'short'", "x")]),
    }
    src = "s = '''line1\nline2'''\n"
    enc = encode_python_source_plan_a(src, books, escape_prefix="__L__")
    assert enc == src
    assert decode_python_source_plan_a(enc, books, escape_prefix="__L__") == src
