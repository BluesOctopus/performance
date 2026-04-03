"""Classifier helpers for semantic free-text string candidates."""

from __future__ import annotations

from dataclasses import dataclass

from ..router import ABRoutingConfig, classify_string_kind


@dataclass(slots=True)
class SemanticClassifierConfig:
    free_text_min_chars: int = 24
    free_text_min_words: int = 4


def is_semantic_free_text(token_spelling: str, cfg: SemanticClassifierConfig) -> bool:
    r = ABRoutingConfig(
        free_text_min_chars=cfg.free_text_min_chars,
        free_text_min_words=cfg.free_text_min_words,
        fallback_unknown=True,
    )
    return classify_string_kind(token_spelling, r) == "B"

