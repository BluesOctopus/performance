"""Hybrid AB config resolution tests."""

from __future__ import annotations

import pytest

from config import (
    STAGE1_HYBRID_AB_AST_MIN_FREQ,
    STAGE1_HYBRID_AB_MIN_TOTAL_NET_SAVING,
    STAGE2_HYBRID_AB_MODE,
    STAGE2_HYBRID_AB_PROFILE,
    STAGE2_PROFILE_FLAGS,
    resolve_hybrid_ab_settings,
)
from eval.v2_eval import eval_mining_cache_name


def test_hybrid_ab_profile_tokenizer_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ET_STAGE3_AB_MODE", raising=False)
    monkeypatch.delenv("ET_STAGE3_AB_ENABLE_B", raising=False)
    monkeypatch.delenv("ET_STAGE3_AB_B_SIMILARITY_THRESHOLD", raising=False)
    g4 = resolve_hybrid_ab_settings("gpt4")
    g2 = resolve_hybrid_ab_settings("gpt2")
    assert g4["mode"] == "exact_only"
    assert g2["mode"] == "exact_only"
    assert g4["enable_b"] is False
    assert g2["enable_b"] is False
    assert g4["b_similarity_threshold"] >= g2["b_similarity_threshold"]
    assert g4["b_risk_threshold"] >= g2["b_risk_threshold"]


def test_hybrid_ab_new_knobs_are_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ET_STAGE3_AB_MODE", "hybrid")
    monkeypatch.setenv("ET_STAGE3_AB_ENABLE_B", "1")
    monkeypatch.setenv("ET_STAGE3_AB_ENABLE_MID_FREE_TEXT", "1")
    monkeypatch.setenv("ET_STAGE3_AB_ALLOW_MULTILINE_WHITELIST", "1")
    monkeypatch.setenv("ET_STAGE3_AB_B_SIMILARITY_KIND", "hybrid_lexical_char")
    monkeypatch.setenv("ET_STAGE3_AB_B_LEXICAL_WEIGHT", "0.6")
    monkeypatch.setenv("ET_STAGE3_AB_B_CHAR_WEIGHT", "0.4")
    monkeypatch.setenv("ET_STAGE3_AB_B_CHAR_NGRAM_N", "4")
    cfg = resolve_hybrid_ab_settings("gpt4")
    assert cfg["mode"] == "hybrid"
    assert cfg["enable_b"] is True
    assert cfg["enable_mid_free_text"] is True
    assert cfg["allow_multiline_whitelist"] is True
    assert cfg["b_similarity_kind"] == "hybrid_lexical_char"
    assert cfg["b_lexical_weight"] == 0.6
    assert cfg["b_char_weight"] == 0.4
    assert cfg["b_char_ngram_n"] == 4


def test_hybrid_ab_stage1_stage2_profile_constants() -> None:
    assert "stage2_hybrid_ab_aggressive" in STAGE2_PROFILE_FLAGS
    assert STAGE2_HYBRID_AB_PROFILE == "stage2_hybrid_ab_aggressive"
    assert STAGE2_HYBRID_AB_MODE == "blockwise"
    assert STAGE1_HYBRID_AB_AST_MIN_FREQ <= 20
    assert STAGE1_HYBRID_AB_MIN_TOTAL_NET_SAVING == 0


def test_eval_mining_cache_name_hybrid_auto_distinct() -> None:
    a = eval_mining_cache_name(80, "hybrid_ab", None, None)
    assert "s2_hybrid_ab_auto" in a
    b = eval_mining_cache_name(80, "legacy", None, None)
    assert "s2_hybrid_ab_auto" not in b

