"""Tokenizer resolution helpers for offline diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_TOKENIZER_NAME = "gpt4"
QWEN_15B_TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-1.5B"


class TokenizerResolutionError(RuntimeError):
    """Raised when the requested tokenizer cannot be resolved."""


class GPT4oTokenizerResolutionError(TokenizerResolutionError):
    """Backward-compatible alias for older evaluation helpers."""


@dataclass(frozen=True)
class ResolvedTokenizer:
    tokenizer_name: str
    encoder: Any
    tok_type: str


def normalize_tokenizer_name(name: str | None) -> str:
    raw = (name or DEFAULT_TOKENIZER_NAME).strip()
    lowered = raw.lower()
    if lowered in {"gpt4", "gpt4o", "gpt-4o", "o200k", "o200k_base"}:
        return "gpt4"
    if lowered in {"cl100k", "cl100k_base"}:
        return "cl100k_base"
    if lowered in {
        "qwen",
        "qwen2.5",
        "qwen2.5-coder",
        "qwen2.5-coder-1.5b",
        "qwen/qwen2.5-coder-1.5b",
    }:
        return QWEN_15B_TOKENIZER_NAME
    return raw


def resolve_tokenizer(name: str | None = None) -> ResolvedTokenizer:
    tokenizer_name = normalize_tokenizer_name(name)
    if tokenizer_name == "gpt4":
        return ResolvedTokenizer(
            tokenizer_name="gpt4",
            encoder=_resolve_tiktoken_model("gpt-4o", "o200k_base"),
            tok_type="tiktoken",
        )
    if tokenizer_name == "cl100k_base":
        return ResolvedTokenizer(
            tokenizer_name="cl100k_base",
            encoder=_resolve_tiktoken_encoding("cl100k_base"),
            tok_type="tiktoken",
        )
    if tokenizer_name == QWEN_15B_TOKENIZER_NAME:
        return ResolvedTokenizer(
            tokenizer_name=QWEN_15B_TOKENIZER_NAME,
            encoder=_resolve_hf_tokenizer(QWEN_15B_TOKENIZER_NAME),
            tok_type="hf",
        )
    raise TokenizerResolutionError(
        f"Unsupported tokenizer {name!r}. Expected one of: gpt4, cl100k_base, {QWEN_15B_TOKENIZER_NAME}."
    )


def count_tokens(text: str, *, encoder: Any, tok_type: str) -> int:
    if tok_type == "tiktoken":
        try:
            return len(encoder.encode(text, allowed_special="all"))
        except TypeError:
            return len(encoder.encode(text))
    try:
        return len(encoder.encode(text, add_special_tokens=False))
    except TypeError:
        return len(encoder.encode(text))


def resolve_gpt4o_base_tokenizer() -> Any:
    return resolve_tokenizer("gpt4").encoder


def count_gpt4o_base_tokens(text: str, *, encoder: Any | None = None) -> int:
    enc = encoder if encoder is not None else resolve_gpt4o_base_tokenizer()
    return count_tokens(text, encoder=enc, tok_type="tiktoken")


def _resolve_tiktoken_model(model_name: str, fallback_encoding: str) -> Any:
    try:
        import tiktoken
    except ImportError as e:
        raise GPT4oTokenizerResolutionError(
            "tiktoken is not installed. Install with: pip install tiktoken"
        ) from e

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return _resolve_tiktoken_encoding(fallback_encoding)


def _resolve_tiktoken_encoding(encoding_name: str) -> Any:
    try:
        import tiktoken
    except ImportError as e:
        raise GPT4oTokenizerResolutionError(
            "tiktoken is not installed. Install with: pip install tiktoken"
        ) from e

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        raise GPT4oTokenizerResolutionError(
            f"Could not load tiktoken encoding {encoding_name!r}: {e!r}"
        ) from e


def _resolve_hf_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise TokenizerResolutionError(
            "transformers is not installed. Install with: pip install transformers"
        ) from e

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise TokenizerResolutionError(
            f"Could not load Hugging Face tokenizer {model_name!r}: {e!r}"
        ) from e
