"""Shared lexical routing rules for Stage3 hybrid AB."""

from __future__ import annotations

import re

RE_PATH_LIKE = re.compile(r"[/\\]|\.py$|\.json$|\.ya?ml$|\.toml$|\.ini$", re.I)
RE_IDENTIFIER_LIKE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
RE_REGEX_LIKE = re.compile(
    r"(\\b|\\d\+|\\s\*|\[\^\\w\\s\]|\(\?:|\.\*|\[[A-Za-z0-9_^-]+\][+*?])"
)
RE_URL = re.compile(r"^https?://", re.I)

