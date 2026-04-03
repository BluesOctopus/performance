"""Unified placeholder-aware sequence counting and vocab introduction cost."""

from __future__ import annotations

import re
from typing import Any, Optional

from config import (
    FIXED_VOCAB_TOKEN_COST,
    PLACEHOLDER_PATTERNS,
    VOCAB_COST_MODE,
)

# Compiled from config (single source of truth for placeholder matching in counting).
PLACEHOLDER_RE = re.compile("|".join(f"(?:{p})" for p in PLACEHOLDER_PATTERNS))


def iter_placeholder_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, token_text) for each placeholder match in order."""
    return [(m.start(), m.end(), m.group(0)) for m in PLACEHOLDER_RE.finditer(text)]


def extract_placeholders(text: str) -> list[str]:
    """All placeholder matches in order (may repeat)."""
    return [m.group(0) for m in PLACEHOLDER_RE.finditer(text)]


def extract_unique_placeholders(text: str) -> list[str]:
    """Unique placeholders in first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for p in extract_placeholders(text):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def split_text_by_placeholders(text: str) -> list[tuple[str, bool]]:
    """Segments: (fragment, is_placeholder)."""
    if not text:
        return []
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in PLACEHOLDER_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def count_base_tokens(
    text: str,
    *,
    tokenizer: Any = None,
    tok_type: Optional[str] = None,
) -> int:
    """Count tokens in *text* with no placeholder special-casing (raw encode length)."""
    if not text:
        return 0
    if tokenizer is None or tok_type is None:
        return len(text.split())
    import marker_count as _mc

    return len(_mc.encode(tokenizer, tok_type, text))


def count_sequence_tokens(
    text: str,
    *,
    tokenizer: Any = None,
    tok_type: Optional[str] = None,
) -> int:
    """Each placeholder counts as 1 token; other spans use ``count_base_tokens``."""
    if not text:
        return 0
    total = 0
    for frag, is_ph in split_text_by_placeholders(text):
        if is_ph:
            total += 1
        else:
            total += count_base_tokens(frag, tokenizer=tokenizer, tok_type=tok_type)
    return total


def serialize_vocab_entry(entry: dict) -> str:
    """Serialize a vocab entry for cost estimation."""
    token = entry.get("token", "")
    definition = entry.get("definition", "")
    return f"{token} => {definition}"


def compute_vocab_intro_cost(
    vocab_entries: list[dict],
    *,
    mode: str,
    tokenizer: Any = None,
    tok_type: Optional[str] = None,
) -> int:
    """Cost to introduce *vocab_entries* (caller dedupes for corpus_once)."""
    if not vocab_entries:
        return 0
    if mode == "fixed_per_token":
        return len(vocab_entries) * FIXED_VOCAB_TOKEN_COST
    if mode == "serialized_definition":
        total = 0
        for e in vocab_entries:
            total += count_base_tokens(
                serialize_vocab_entry(e),
                tokenizer=tokenizer,
                tok_type=tok_type,
            )
        return total
    raise ValueError(f"unknown vocab cost mode: {mode}")


def build_plan_a_vocab_entries(
    codebooks: dict,
    *,
    escape_prefix: str,
    tag_by_field: dict[str, str] | None = None,
) -> list[dict]:
    """
    Full-codebook vocab entries (legacy helper; prefer :func:`build_used_plan_a_vocab_entries`).
    """
    tag_by_field = tag_by_field or {
        "variable": "V",
        "attribute": "A",
        "string": "S",
    }
    entries: list[dict] = []
    for field, cb in codebooks.items():
        tag = tag_by_field.get(field)
        if tag is None:
            continue
        assignments = getattr(cb, "assignments", []) or []
        for a in assignments:
            literal = getattr(a, "literal", "")
            code = getattr(a, "code", "")
            token = f"{escape_prefix}{tag}{code}"
            entries.append(
                {
                    "token": token,
                    "kind": "stage3_plan_a",
                    "definition": repr(literal),
                }
            )
    return entries


def build_used_plan_a_vocab_entries(
    codebooks: dict,
    used: set[tuple[str, str]],
    *,
    escape_prefix: str,
) -> list[dict]:
    """
    Vocab intro entries for **actually used** (field, code) pairs only.

    *definition* uses the literal spelling from the codebook (same string as in source
    for STRING tokens, e.g. ``'\"x\"'``).
    *token* is the real compressed surface form (NAME or STRING token text).
    """
    from stage3.literal_codec.pipeline.surface_cost import (
        surface_text_for_code,
    )

    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for field, code in sorted(used):
        if (field, code) in seen:
            continue
        seen.add((field, code))
        cb = codebooks.get(field)
        if cb is None:
            continue
        literal = None
        for a in getattr(cb, "assignments", []) or []:
            if getattr(a, "code", None) == code:
                literal = getattr(a, "literal", "")
                break
        if literal is None:
            continue
        surface = surface_text_for_code(field, code, escape_prefix)
        entries.append(
            {
                "token": surface,
                "kind": "stage3_plan_a_used",
                "definition": literal,
                "field": field,
                "code": code,
            }
        )
    return entries


def compute_effective_total_tokens(
    text: str,
    vocab_entries: list[dict],
    *,
    vocab_cost_mode: str = VOCAB_COST_MODE,
    tokenizer: Any = None,
    tok_type: Optional[str] = None,
) -> dict[str, int]:
    seq = count_sequence_tokens(text, tokenizer=tokenizer, tok_type=tok_type)
    vocab_intro = compute_vocab_intro_cost(
        vocab_entries,
        mode=vocab_cost_mode,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )
    return {
        "sequence_only_tokens": seq,
        "vocab_intro_tokens": vocab_intro,
        "effective_total_tokens": seq + vocab_intro,
    }
