"""Hybrid AB config resolution tests."""

from __future__ import annotations

import pytest

from config import resolve_hybrid_ab_settings


def test_hybrid_ab_profile_tokenizer_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ET_STAGE3_AB_B_SIMILARITY_THRESHOLD", raising=False)
    g4 = resolve_hybrid_ab_settings("gpt4")
    g2 = resolve_hybrid_ab_settings("gpt2")
    assert g4["b_similarity_threshold"] >= g2["b_similarity_threshold"]
    assert g4["b_risk_threshold"] >= g2["b_risk_threshold"]

