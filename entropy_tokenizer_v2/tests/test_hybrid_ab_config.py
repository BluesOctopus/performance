"""Hybrid AB config resolution tests."""

from __future__ import annotations

import pytest

from config import resolve_hybrid_ab_settings


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

