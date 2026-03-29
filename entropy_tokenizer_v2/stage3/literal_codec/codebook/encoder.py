"""Field value encoder."""

from __future__ import annotations

from dataclasses import dataclass

from literal_codec.types import FieldCodebook


@dataclass(slots=True)
class FieldEncoder:
    """Encode a literal with a field codebook."""

    codebook: FieldCodebook
    _literal_to_code: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._literal_to_code = {item.literal: item.code for item in self.codebook.assignments}

    def encode_value(self, value: str) -> str:
        escape = self.codebook.escape_prefix
        if value in self._literal_to_code:
            return f"{escape}{self._literal_to_code[value]}"
        if value.startswith(escape):
            return f"{escape}{value}"
        return value
