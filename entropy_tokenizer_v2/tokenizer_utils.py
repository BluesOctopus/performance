"""Local GPT-4o / o200k base token counting (no remote API)."""

from __future__ import annotations

from typing import Any


class GPT4oTokenizerResolutionError(RuntimeError):
    """Raised when neither gpt-4o nor o200k_base can be loaded."""


def resolve_gpt4o_base_tokenizer() -> Any:
    """
    Prefer ``tiktoken.encoding_for_model("gpt-4o")``, fall back to ``o200k_base``.

    Returns a tiktoken ``Encoding`` with ``encode(text)`` for raw base-token counts.
    """
    try:
        import tiktoken
    except ImportError as e:
        raise GPT4oTokenizerResolutionError(
            "tiktoken is not installed. Install with: pip install tiktoken\n"
            "This project uses tiktoken for local GPT-4o/o200k_base token counts."
        ) from e

    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except Exception as e1:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception as e2:
            raise GPT4oTokenizerResolutionError(
                "Could not load GPT-4o encoding or o200k_base fallback.\n"
                f"  encoding_for_model('gpt-4o'): {e1!r}\n"
                f"  get_encoding('o200k_base'): {e2!r}\n"
                "Upgrade tiktoken: pip install -U tiktoken"
            ) from e2


def count_gpt4o_base_tokens(text: str, *, encoder: Any | None = None) -> int:
    """Raw o200k/GPT-4o base token count (not placeholder-aware)."""
    enc = encoder if encoder is not None else resolve_gpt4o_base_tokenizer()
    return len(enc.encode(text))
