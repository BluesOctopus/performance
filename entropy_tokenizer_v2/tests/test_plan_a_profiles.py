"""Tokenizer-aware Plan A profile resolution."""

from __future__ import annotations

import os

import pytest

from config import plan_a_profile_name_for_tokenizer, resolve_plan_a_settings


def test_gpt4_defaults_to_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ET_STAGE3_PLAN_A_PROFILE", raising=False)
    assert plan_a_profile_name_for_tokenizer("gpt4") == "gpt4_conservative"
    r = resolve_plan_a_settings("gpt4")
    assert r["profile_name"] == "gpt4_conservative"
    assert r["min_gain"] == 0.003
    assert r["max_assignments"]["string"] == 16
    assert r["string_filter"]["strict_heuristics"] is True


def test_gpt2_defaults_preserve_wide_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ET_STAGE3_PLAN_A_PROFILE", raising=False)
    assert plan_a_profile_name_for_tokenizer("gpt2") == "default"
    r = resolve_plan_a_settings("gpt2")
    assert r["min_gain"] == 0.001
    assert r["max_assignments"]["variable"] == 256


def test_explicit_profile_va_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ET_STAGE3_PLAN_A_PROFILE", "gpt4_va_only")
    r = resolve_plan_a_settings("gpt4")
    assert "string" not in r["enabled_categories"]
    assert r["string_filter"] is None
