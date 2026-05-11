"""Routing helpers for Stage3 hybrid A/B literal processing."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .rules import RE_IDENTIFIER_LIKE, RE_PATH_LIKE, RE_REGEX_LIKE, RE_URL


@dataclass(slots=True)
class ABRoutingConfig:
    """Routing knobs for literal channels."""

    free_text_min_chars: int = 24
    free_text_min_words: int = 4
    fallback_unknown: bool = True
    short_string_policy: str = "exact_candidate"
    # Fully configurable key-like regex patterns; default is empty (no implicit A).
    key_like_patterns: tuple[str, ...] = ()
    # Optional relaxed lane for medium-length free-text.
    free_text_mid_min_chars: int = 14
    free_text_mid_min_words: int = 3
    enable_mid_free_text: bool = False
    # Optional multiline lane (disabled by default for safety).
    allow_multiline_whitelist: bool = False
    multiline_max_lines: int = 3
    multiline_max_chars: int = 220


def _inner_string(token_spelling: str) -> str | None:
    try:
        v = ast.literal_eval(token_spelling)
    except (SyntaxError, ValueError, MemoryError):
        return None
    return v if isinstance(v, str) else None


def route_name_literal(field: str) -> str:
    """Name literals are exact by default."""
    if field in {"variable", "attribute"}:
        return "A"
    return "fallback"


def classify_string_kind(token_spelling: str, cfg: ABRoutingConfig) -> str:
    return classify_string_with_reason(token_spelling, cfg)[0]


def classify_string_with_reason(token_spelling: str, cfg: ABRoutingConfig) -> tuple[str, str]:
    inner = _inner_string(token_spelling)
    if inner is None:
        return "fallback", "bad_literal"
    s = inner.strip()
    if not s:
        return "fallback", "empty"
    is_multiline = "\n" in s or "\r" in s
    if is_multiline:
        if not cfg.allow_multiline_whitelist:
            return "fallback", "multiline_disabled"
        lines = [ln for ln in s.replace("\r\n", "\n").split("\n") if ln.strip()]
        if not lines:
            return "fallback", "multiline_empty"
        if len(lines) > max(1, int(cfg.multiline_max_lines)):
            return "fallback", "multiline_too_many_lines"
        if len(s) > max(1, int(cfg.multiline_max_chars)):
            return "fallback", "multiline_too_long"
        # Keep multiline path conservative: only let clearly sentence-like text through.
        words_ml = re.findall(r"[A-Za-z][A-Za-z0-9_]+", s)
        if len(words_ml) < max(2, int(cfg.free_text_mid_min_words)):
            return "fallback", "multiline_not_sentence_like"
        return "B", "multiline_whitelist"
    if RE_URL.search(s) or RE_PATH_LIKE.search(s):
        return "A", "path_or_url"
    if RE_REGEX_LIKE.search(s):
        return "A", "regex_like"
    if RE_IDENTIFIER_LIKE.match(s) and 3 <= len(s) <= 80:
        return "A", "identifier_like"
    for p in cfg.key_like_patterns:
        try:
            if re.search(p, s, re.I):
                return "A", "key_like"
        except re.error:
            continue
    if len(s) <= 2:
        if cfg.short_string_policy == "fallback":
            return "fallback", "short_literal"
        return "A", "short_literal"
    words = s.split()
    punct = sum(1 for ch in s if not ch.isalnum() and not ch.isspace())
    if (
        len(s) >= cfg.free_text_min_chars
        and len(words) >= cfg.free_text_min_words
        and " " in s
        and punct / max(1, len(s)) < 0.22
    ):
        return "B", "free_text"
    if (
        cfg.enable_mid_free_text
        and len(s) >= max(3, int(cfg.free_text_mid_min_chars))
        and len(words) >= max(2, int(cfg.free_text_mid_min_words))
        and " " in s
        and punct / max(1, len(s)) < 0.18
    ):
        return "B", "mid_free_text"
    return ("fallback", "unknown") if cfg.fallback_unknown else ("A", "fallback_to_exact")

