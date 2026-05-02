"""Shared marker helpers for Stage1 placeholder recognition."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from placeholder_accounting import PLACEHOLDER_RE

if TYPE_CHECKING:
    from marker_optimizer import MarkerScheme


RE_ALL_MARKERS = PLACEHOLDER_RE
RE_SYN_ONLY = re.compile(r"<SYN_\d+>")
RE_SYN_LINE = re.compile(r"^\s*<SYN_\d+>(?:\s|$)")
RE_SYN_MARKER_EXACT = re.compile(r"^<SYN_\d+>$")

_ACTIVE_MARKER_SCHEME: MarkerScheme | None = None


def set_active_marker_scheme(scheme: MarkerScheme | None) -> None:
    global _ACTIVE_MARKER_SCHEME
    _ACTIVE_MARKER_SCHEME = scheme


def get_active_marker_scheme() -> MarkerScheme | None:
    return _ACTIVE_MARKER_SCHEME


def make_syn_marker(index: int, scheme: MarkerScheme | None = None) -> str:
    active = scheme or _ACTIVE_MARKER_SCHEME
    if active is not None and 0 <= index < len(active.markers):
        return active.marker(index)
    return f"<SYN_{index}>"


def is_syn_marker(text: str, scheme: MarkerScheme | None = None) -> bool:
    active = scheme or _ACTIVE_MARKER_SCHEME
    if active is not None:
        return text in set(active.markers)
    return bool(RE_SYN_MARKER_EXACT.match(text))


def is_syn_line(line: str, scheme: MarkerScheme | None = None) -> bool:
    marker = extract_line_marker(line, scheme=scheme)
    return marker is not None


def extract_line_marker(line: str, scheme: MarkerScheme | None = None) -> str | None:
    stripped = line.lstrip()
    if not stripped:
        return None
    token = stripped.split(maxsplit=1)[0]
    return token if is_syn_marker(token, scheme=scheme) else None


def get_syn_line_spans(text: str, scheme: MarkerScheme | None = None) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        if is_syn_line(line, scheme=scheme):
            spans.append((offset, end))
        offset = end
    return spans


def extract_stage1_markers(text: str, scheme: MarkerScheme | None = None) -> list[str]:
    active = scheme or _ACTIVE_MARKER_SCHEME
    if active is None:
        return [match.group(0) for match in RE_SYN_ONLY.finditer(text)]
    found: list[str] = []
    markers = sorted(active.markers, key=len, reverse=True)
    for line in text.splitlines():
        token = extract_line_marker(line, scheme=active)
        if token is not None:
            found.append(token)
            continue
        for marker in markers:
            if marker in line:
                found.append(marker)
    return found


def is_placeholder_token(text: str) -> bool:
    return bool(text and PLACEHOLDER_RE.fullmatch(text))
