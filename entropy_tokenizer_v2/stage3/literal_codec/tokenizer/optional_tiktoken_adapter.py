"""Optional tiktoken adapter with graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass

from literal_codec.tokenizer.base import TokenizerAdapter
from literal_codec.tokenizer.mock_tokenizer import MockTokenizerAdapter


@dataclass(slots=True)
class OptionalTiktokenAdapter(TokenizerAdapter):
    """
    Adapter that uses tiktoken when available, otherwise falls back to mock tokenizer.
    """

    model_name: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        self._fallback = MockTokenizerAdapter()
        self._enc = None
        try:
            import tiktoken  # type: ignore

            self._enc = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            self._enc = None

    @property
    def is_fallback(self) -> bool:
        return self._enc is None

    def tokenize(self, text: str) -> list[str]:
        if self._enc is None:
            return self._fallback.tokenize(text)
        token_ids = self._enc.encode(text)
        return [str(tid) for tid in token_ids]

    def token_length(self, text: str) -> int:
        if self._enc is None:
            return self._fallback.token_length(text)
        return len(self._enc.encode(text))
