"""Tokenizer adapter interfaces."""

from __future__ import annotations

from typing import Protocol


class TokenizerAdapter(Protocol):
    """Abstract tokenizer API used by the optimization pipeline."""

    def token_length(self, text: str) -> int:
        """Return number of tokens for input text."""

    def tokenize(self, text: str) -> list[str]:
        """Return token list for input text."""
