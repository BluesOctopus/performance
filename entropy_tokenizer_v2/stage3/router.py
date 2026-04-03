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
    # Fully configurable key-like regex patterns; default is empty (no implicit A).
    key_like_patterns: tuple[str, ...] = ()


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
    """
    Route one STRING token spelling to A/B/fallback.

    - A: exact-like string (path/regex/url/identifier/key-like)
    - B: free-text natural-language string
    - fallback: unknown / unsafe
    """
    inner = _inner_string(token_spelling)
    if inner is None:
        return "fallback"
    s = inner.strip()
    if not s:
        return "fallback"
    if "\n" in s or "\r" in s:
        return "fallback"
    if RE_URL.search(s) or RE_PATH_LIKE.search(s):
        return "A"
    if RE_REGEX_LIKE.search(s):
        return "A"
    if RE_IDENTIFIER_LIKE.match(s) and 3 <= len(s) <= 80:
        return "A"
    for p in cfg.key_like_patterns:
        try:
            if re.search(p, s, re.I):
                return "A"
        except re.error:
            continue
    if len(s) <= 2:
        # Short literals are allowed to enter exact aliasing (A); the final
        # gain decision is handled inside alias codec.
        return "A"
    words = s.split()
    # free-text signal: enough words + has whitespace + not symbol-heavy
    punct = sum(1 for ch in s if not ch.isalnum() and not ch.isspace())
    if (
        len(s) >= cfg.free_text_min_chars
        and len(words) >= cfg.free_text_min_words
        and " " in s
        and punct / max(1, len(s)) < 0.22
    ):
        return "B"
    return "fallback" if cfg.fallback_unknown else "A"


def classify_string_with_reason(token_spelling: str, cfg: ABRoutingConfig) -> tuple[str, str]:
    """Return (route, reason) for diagnostics/tests."""
    inner = _inner_string(token_spelling)
    if inner is None:
        return "fallback", "bad_literal"
    s = inner.strip()
    if not s:
        return "fallback", "empty"
    if "\n" in s or "\r" in s:
        return "fallback", "multiline"
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
    return ("fallback", "unknown") if cfg.fallback_unknown else ("A", "fallback_to_exact")
