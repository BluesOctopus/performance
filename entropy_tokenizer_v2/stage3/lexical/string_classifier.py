"""Classifier helpers for lexical free-text string candidates."""

from __future__ import annotations

from dataclasses import dataclass

from ..routing.router import ABRoutingConfig, classify_string_with_reason


@dataclass(slots=True)
class SemanticClassifierConfig:
    free_text_min_chars: int = 24
    free_text_min_words: int = 4
    enable_mid_free_text: bool = False
    free_text_mid_min_chars: int = 14
    free_text_mid_min_words: int = 3
    allow_multiline_whitelist: bool = False
    multiline_max_lines: int = 3
    multiline_max_chars: int = 220


def is_semantic_free_text(token_spelling: str, cfg: SemanticClassifierConfig) -> bool:
    route, _reason = classify_semantic_free_text(token_spelling, cfg)
    return route == "B"


def classify_semantic_free_text(
    token_spelling: str,
    cfg: SemanticClassifierConfig,
) -> tuple[str, str]:
    r = ABRoutingConfig(
        free_text_min_chars=cfg.free_text_min_chars,
        free_text_min_words=cfg.free_text_min_words,
        fallback_unknown=True,
        short_string_policy="fallback",
        enable_mid_free_text=cfg.enable_mid_free_text,
        free_text_mid_min_chars=cfg.free_text_mid_min_chars,
        free_text_mid_min_words=cfg.free_text_mid_min_words,
        allow_multiline_whitelist=cfg.allow_multiline_whitelist,
        multiline_max_lines=cfg.multiline_max_lines,
        multiline_max_chars=cfg.multiline_max_chars,
    )
    return classify_string_with_reason(token_spelling, r)

