"""Field value decoder."""

from __future__ import annotations

from dataclasses import dataclass

from literal_codec.exceptions import DecodeError
from literal_codec.types import FieldCodebook


@dataclass(slots=True)
class FieldDecoder:
    """Decode a value produced by FieldEncoder."""

    codebook: FieldCodebook
    strict: bool = True
    _code_to_literal: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._code_to_literal = {item.code: item.literal for item in self.codebook.assignments}

    def decode_value(self, value: str) -> str:
        escape = self.codebook.escape_prefix
        if not value.startswith(escape):
            return value

        payload = value[len(escape) :]
        if payload.startswith(escape):
            return payload

        if payload in self._code_to_literal:
            return self._code_to_literal[payload]

        if self.strict:
            raise DecodeError(f"Unknown code payload: {payload}")
        return value
