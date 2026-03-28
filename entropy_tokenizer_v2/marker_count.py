"""Count tokenizer ids as if each placeholder were one token (unified accounting)."""

from __future__ import annotations

from re import Pattern

from markers import RE_ALL_MARKERS
from placeholder_accounting import count_sequence_tokens
from markers import is_syn_marker


def encode(tokenizer, tok_type: str, text: str) -> list[int]:
    if tok_type == "tiktoken":
        return tokenizer.encode(text, allowed_special="all")
    return tokenizer.encode(text, add_special_tokens=False)


def count_augmented(
    text: str,
    tokenizer,
    tok_type: str,
    *,
    pattern: Pattern[str] = RE_ALL_MARKERS,
) -> int:
    del pattern
    return count_sequence_tokens(text, tokenizer=tokenizer, tok_type=tok_type)


def count_augmented_text_fragment(
    text: str,
    tokenizer,
    tok_type: str,
    *,
    pattern: Pattern[str] = RE_ALL_MARKERS,
) -> int:
    del pattern
    return count_sequence_tokens(text, tokenizer=tokenizer, tok_type=tok_type)


def count_syn_marker(marker: str) -> int:
    if not is_syn_marker(marker):
        raise ValueError(f"not a valid SYN marker: {marker}")
    return 1
