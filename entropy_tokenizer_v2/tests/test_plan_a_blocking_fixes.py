"""Blocking-fix regression tests for Stage3 Plan A (mutual exclusivity, surface cost, used vocab)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "stage3"
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))

from literal_codec.pipeline.source_mining import collect_category_literal_streams
from literal_codec.pipeline.surface_cost import encoded_form_token_cost, surface_text_for_code
from literal_codec.pipeline.source_codec import (
    _encode_string_token,
    extract_used_plan_a_entries,
)
from literal_codec.tokenizer.mock_tokenizer import MockTokenizerAdapter
from literal_codec.types import CodeAssignment, FieldCodebook, FieldProfile, LiteralStat

from placeholder_accounting import build_plan_a_vocab_entries, build_used_plan_a_vocab_entries


def test_source_mining_attribute_variable_mutually_exclusive():
    src = "class Obj:\n    def m(self):\n        self.attr = value\n"
    streams, diag = collect_category_literal_streams(
        [src],
        enabled_categories=frozenset({"variable", "attribute", "string"}),
    )
    var_toks = set(streams["variable"])
    attr_toks = set(streams["attribute"])
    assert "attr" in attr_toks
    assert "attr" not in var_toks
    assert "value" in var_toks
    assert "self" not in var_toks
    assert diag.variable_occurrences >= 1
    assert diag.attribute_occurrences >= 1


def test_plan_a_cost_model_uses_real_surface_form():
    tok = MockTokenizerAdapter()
    escape = "__L__"
    bare = "a"
    c_bare = tok.token_length(bare)
    c_var = encoded_form_token_cost("variable", bare, escape, tok)
    surf = surface_text_for_code("variable", bare, escape)
    assert surf == f"{escape}V{bare}"
    assert c_var == tok.token_length(surf)
    assert c_bare != c_var or len(surf) == len(bare)


def test_plan_a_string_cost_matches_actual_encoded_token():
    escape = "__L__"
    code = "x"
    assert _encode_string_token(escape, code) == surface_text_for_code("string", code, escape)


def test_plan_a_used_entries_only_vocab_intro():
    escape = "__L__"
    book = FieldCodebook(
        field_name="variable",
        version="t",
        assignments=[
            CodeAssignment("foo", "a", 3, 2, 0.1),
            CodeAssignment("bar", "b", 3, 2, 0.1),
        ],
        escape_prefix=escape,
        metadata={},
    )
    books = {"variable": book}
    text = f"{escape}Va = 1\n"
    used = extract_used_plan_a_entries(text, books, escape)
    assert used == {("variable", "a")}
    e_used = build_used_plan_a_vocab_entries(books, used, escape_prefix=escape)
    e_all = build_plan_a_vocab_entries(books, escape_prefix=escape)
    assert len(e_used) == 1
    assert len(e_all) == 2


def test_extract_used_skips_double_escape():
    escape = "__L__"
    book = FieldCodebook("variable", "t", [], escape, {})
    books = {"variable": book}
    text = f"{escape}{escape}raw = 1\n"
    assert extract_used_plan_a_entries(text, books, escape) == set()
