from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Optional

from placeholder_accounting import compute_vocab_intro_cost
from config import VOCAB_COST_MODE


@dataclass(slots=True)
class Stage3EncodeResult:
    encoded_text: str
    # Entries that feed `compute_vocab_intro_cost`.
    # Shape is token/definition compatible with placeholder_accounting.
    vocab_entries: list[dict[str, Any]]
    # Extra backend-specific metrics for diagnostics (not required for accounting).
    metrics: dict[str, Any]


class Stage3Backend(Protocol):
    name: str

    def encode(
        self,
        text: str,
        repo_config: Any,
        *,
        tokenizer: Any,
        tok_type: Optional[str],
    ) -> Stage3EncodeResult:
        ...

    def compute_intro_cost(
        self,
        result: Stage3EncodeResult,
        *,
        tokenizer: Any,
        tok_type: Optional[str],
    ) -> int:
        return compute_vocab_intro_cost(
            result.vocab_entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )

