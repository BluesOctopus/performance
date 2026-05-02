"""Regex-based mock tokenizer for deterministic local tests."""

from __future__ import annotations

import re

from .base import TokenizerAdapter

TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|_+|[^A-Za-z0-9_\s]")


class MockTokenizerAdapter(TokenizerAdapter):
    """
    A simple tokenizer:
    - alphabetic chunks, numeric chunks and underscore runs as tokens
    - each punctuation symbol as one token
    - whitespace ignored
    """

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    def token_length(self, text: str) -> int:
        return len(self.tokenize(text))
