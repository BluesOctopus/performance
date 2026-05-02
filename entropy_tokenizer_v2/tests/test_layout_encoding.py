"""Reversible layout indentation encoding (<NL0>–<NL8>)."""

from __future__ import annotations

import pytest

from lossy_cleaner import CleaningStats
from marker_count import count_augmented
from markers import is_placeholder_token
from pipeline import apply_stage1_stage2_adapted, apply_stage1_stage2_layout_safe_experimental
from placeholder_accounting import compute_vocab_intro_cost, count_sequence_tokens
from stage2.config import STAGE2_ADAPTED_ORDER_LABEL, STAGE2_LAYOUT_EXPERIMENTAL_ORDER_LABEL
from stage2.layout_encoding import (
    build_layout_token,
    decode_layout_indentation,
    encode_layout_indentation,
    estimate_layout_encoding_effect,
    infer_indent_unit,
    is_layout_token,
    verify_layout_roundtrip,
)


def test_build_and_detect_layout_token() -> None:
    assert build_layout_token(3) == "<NL3>"
    assert is_layout_token("<NL3>")
    assert not is_layout_token("<NL9>")
    assert not is_layout_token("x")


def test_roundtrip_four_spaces() -> None:
    src = "def f():\n    if 1:\n        return 0\n"
    v = verify_layout_roundtrip(src)
    assert v["roundtrip_equal"] is True
    assert v["encoded_token_count"] >= 1
    enc = encode_layout_indentation(src)
    assert enc["usable"] is True
    dec = decode_layout_indentation(
        enc["encoded_text"],
        indent_unit=int(enc["indent_unit"] or 1),
        indent_style=str(enc["indent_style"]),
    )
    assert dec == src


def test_roundtrip_two_spaces() -> None:
    src = "def g():\n  pass\n"
    assert verify_layout_roundtrip(src)["roundtrip_equal"] is True


def test_mixed_tabs_spaces_not_usable() -> None:
    src = "a\n\t x\n"
    info = infer_indent_unit(src)
    assert info["usable"] is False
    enc = encode_layout_indentation(src)
    assert enc["encoded_text"] == src.replace("\r\n", "\n").replace("\r", "\n")
    assert enc["encoded_line_count"] == 0


def test_multiline_string_indent_preserved() -> None:
    src = (
        's = """\n'
        "    inner line\n"
        "    second\n"
        '"""\n'
        "def f():\n"
        "    return s\n"
    )
    v = verify_layout_roundtrip(src)
    assert v["roundtrip_equal"] is True
    enc = encode_layout_indentation(src)
    dec = decode_layout_indentation(
        enc["encoded_text"],
        indent_unit=int(enc["indent_unit"] or 1),
        indent_style=str(enc["indent_style"]),
    )
    assert dec == src


def test_placeholder_aware_counting_layout_token() -> None:
    class SplitTok:
        def encode(self, text: str, allowed_special: str = "all", **kwargs):
            del allowed_special, kwargs
            return text.split()

    tok = SplitTok()
    assert count_sequence_tokens("<NL2>", tokenizer=tok, tok_type="hf") == 1
    assert count_sequence_tokens("hello <NL0> world", tokenizer=tok, tok_type="hf") == 3


def test_is_placeholder_token_includes_nl() -> None:
    assert is_placeholder_token("<NL0>")


def test_vocab_intro_cost_for_layout_tokens() -> None:
    entries = [{"token": "<NL0>", "definition": "<NL0>", "kind": "layout"}]
    c = compute_vocab_intro_cost(entries, mode="fixed_per_token", tokenizer=None, tok_type=None)
    assert c > 0


def test_estimate_layout_effect_keys() -> None:
    class Tok:
        def encode(self, text: str, allowed_special: str = "all", **kwargs):
            del allowed_special, kwargs
            return text.split()

    base = "def f():\n    pass\n"
    enc = encode_layout_indentation(base)
    eff = estimate_layout_encoding_effect(
        base,
        enc["encoded_text"],
        enc["layout_tokens_used"],
        tokenizer=Tok(),
        tok_type="hf",
        vocab_cost_mode="fixed_per_token",
    )
    assert "effective_total_net_saving" in eff
    assert "vocab_intro_tokens" in eff


def test_experimental_pipeline_order_distinct_from_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pipeline.apply_stage1_with_stats",
        lambda *_a, **_k: ("x", {}),
    )
    monkeypatch.setattr(
        "pipeline.run_stage2_pre_safe",
        lambda *a, **k: (a[0], CleaningStats()),
    )
    monkeypatch.setattr(
        "pipeline.run_stage2_post_surface",
        lambda *a, **k: (a[0], CleaningStats()),
    )
    monkeypatch.setattr(
        "pipeline.run_stage2_post_layout_encode",
        lambda text, **_: (text, {"layout_encoding_used": False}),
    )
    monkeypatch.setattr("pipeline._count_with_ops", lambda *_a, **_k: 1)
    monkeypatch.setattr("pipeline._stage1_vocab_intro", lambda *_a, **_k: 0)

    class R:
        def skeleton_candidates(self):
            return []

    out_safe = apply_stage1_stage2_adapted(
        "def f():\n  pass\n",
        R(),
        stage2_profile="safe",
        tokenizer=None,
        tok_type="x",
    )
    out_lo = apply_stage1_stage2_layout_safe_experimental(
        "def f():\n  pass\n",
        R(),
        tokenizer=None,
        tok_type="x",
    )
    assert out_safe["stage2_order"] == STAGE2_ADAPTED_ORDER_LABEL
    assert out_lo["stage2_order"] == STAGE2_LAYOUT_EXPERIMENTAL_ORDER_LABEL
    assert out_safe["stage2_profile"] == "safe"
    assert out_lo["stage2_profile"] == "layout_safe_experimental"


def test_count_augmented_accepts_nl_markers() -> None:
    try:
        import tiktoken
    except ImportError:
        pytest.skip("tiktoken not installed")
    enc = tiktoken.get_encoding("o200k_base")
    assert count_augmented("<NL4>", enc, "tiktoken") == 1
