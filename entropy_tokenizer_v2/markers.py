"""Shared marker regexes and detection helpers."""

from __future__ import annotations

import re

from placeholder_accounting import PLACEHOLDER_RE

# Back-compat: same as placeholder union used for counting.
RE_ALL_MARKERS = PLACEHOLDER_RE
RE_SYN_ONLY = re.compile(r"<SYN_\d+>")

# "<SYN_N>" must appear at line start (optional leading spaces),
# then either whitespace or end-of-line.
RE_SYN_LINE = re.compile(r"^\s*<SYN_\d+>(?:\s|$)")
RE_SYN_MARKER_EXACT = re.compile(r"^<SYN_\d+>$")


def make_syn_marker(index: int) -> str:
    return f"<SYN_{index}>"


def is_syn_marker(text: str) -> bool:
    return bool(RE_SYN_MARKER_EXACT.match(text))


def is_syn_line(line: str) -> bool:
    return bool(RE_SYN_LINE.match(line))


def is_placeholder_token(text: str) -> bool:
    """True if *text* is a single placeholder token (SYN / VAR / ATTR / …)."""
    return bool(text and PLACEHOLDER_RE.fullmatch(text))


def get_syn_line_spans(text: str) -> list[tuple[int, int]]:
    """Character spans for lines that start with ``<SYN_N>``."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        if is_syn_line(line):
            spans.append((offset, end))
        offset = end
    return spans
