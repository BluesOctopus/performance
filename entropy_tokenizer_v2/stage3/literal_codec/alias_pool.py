"""Legal Python identifier pool for Stage3 hybrid A-channel aliases.

Why: hand-built families like ``__ab0`` often split into multiple cl100k_base tokens,
so we prefer real identifiers whose encoded length is 1–2 tokens under the active tokenizer.
"""

from __future__ import annotations

import builtins
import keyword
from typing import Any

_BASE_BLOCKED = (
    set(keyword.kwlist)
    | {n for n in dir(builtins) if isinstance(n, str)}
    | {"self", "cls", "True", "False", "None", "Ellipsis", "NotImplemented"}
)


def is_legal_public_identifier(name: str) -> bool:
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


def _raw_identifier_stream() -> list[str]:
    """Deterministic stream of candidate identifiers (short first)."""
    out: list[str] = []
    for c in "abcdefghijklmnopqrstuvwxyz":
        out.append(c)
    for i in range(512):
        for c in "abcdefghijklmnopqrstuvwxyz":
            out.append(f"{c}{i}")
    for i in range(1000):
        a = i % 26
        b = (i // 26) % 26
        c = (i // 676) % 26
        out.append(f"{chr(ord('a') + a)}{chr(ord('a') + b)}{chr(ord('a') + c)}")
    return out


def build_legal_alias_alphabet(
    tokenizer: Any,
    tok_type: str,
    *,
    reserved: set[str],
    max_n: int = 256,
    max_alias_token_len: int = 2,
) -> list[str]:
    """
    Return aliases that are legal identifiers, not keywords/builtins/conflicts,
    sorted by tokenizer token count (then length, then lexicographic).
    """
    from marker_count import encode as mc_encode

    def tok_len(surface: str) -> int:
        return len(mc_encode(tokenizer, tok_type, surface))

    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for name in _raw_identifier_stream():
        if name in seen:
            continue
        if not is_legal_public_identifier(name):
            continue
        if name in _BASE_BLOCKED or name in reserved:
            continue
        if name.startswith("__"):
            continue
        tl = tok_len(name)
        if tl > max_alias_token_len:
            continue
        seen.add(name)
        scored.append((tl, len(name), name))
        if len(scored) >= max_n * 4:
            break
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[2] for t in scored[:max_n]]
