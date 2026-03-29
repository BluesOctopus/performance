"""Bridge entropy_tokenizer_v2 tokenizer to literal_codec TokenizerAdapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from literal_codec.tokenizer.base import TokenizerAdapter


@dataclass(slots=True)
class V2TokenizerAdapter(TokenizerAdapter):
    """Token length matches marker_count.encode (same as v2 pipeline counting)."""

    tokenizer: Any
    tok_type: str

    def token_length(self, text: str) -> int:
        from marker_count import encode

        return len(encode(self.tokenizer, self.tok_type, text))

    def tokenize(self, text: str) -> list[str]:
        from marker_count import encode

        ids = encode(self.tokenizer, self.tok_type, text)
        return [str(i) for i in ids]
