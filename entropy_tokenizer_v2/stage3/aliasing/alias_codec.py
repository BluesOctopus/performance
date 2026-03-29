"""Request-local exact aliasing for Stage3 hybrid A channel."""

from __future__ import annotations

import io
import keyword
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import builtins

from placeholder_accounting import compute_vocab_intro_cost
from token_scorer import _line_start_offsets, _pos_to_offset

from router import ABRoutingConfig, classify_string_kind
from config import VOCAB_COST_MODE

_PROTECTED = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls", "True", "False", "None"}


@dataclass(slots=True)
class AEntry:
    field: str
    literal: str
    alias: str
    count: int
    raw_cost: int
    alias_cost: int
    intro_cost: int
    gain: int


@dataclass(slots=True)
class ACodecResult:
    encoded_text: str
    entries: list[AEntry] = field(default_factory=list)
    candidates: int = 0
    selected: int = 0
    used_entries: int = 0
    intro_tokens: int = 0
    sequence_saved: int = 0
    effective_net_saving: int = 0
    vocab_entries: list[dict[str, Any]] = field(default_factory=list)


def _token_len(tokenizer: Any, tok_type: str, text: str) -> int:
    from marker_count import encode as _encode

    return len(_encode(tokenizer, tok_type, text))


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    out = text
    for st, ed, rep in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:st] + rep + out[ed:]
    return out


def _build_alias_pool(n: int) -> list[str]:
    out: list[str] = []
    i = 0
    while len(out) < n:
        out.append(f"__ab{i}")
        out.append(f"_a{i}")
        out.append(f"a{i}")
        i += 1
    return out[:n]


def _choose_aliases(alias_pool: list[str], tokenizer: Any, tok_type: str) -> list[str]:
    return sorted(alias_pool, key=lambda x: (_token_len(tokenizer, tok_type, x), len(x), x))


def encode_exact_aliases(
    text: str,
    *,
    tokenizer: Any,
    tok_type: str,
    route_cfg: ABRoutingConfig,
) -> ACodecResult:
    """Encode variable/attribute + exact-like string literals using local aliases."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ACodecResult(encoded_text=text)
    line_starts = _line_start_offsets(text)
    occ: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    prev_is_dot = False
    for tok in toks:
        ttype, tstr = tok.type, tok.string
        if ttype == tokenize.NAME:
            if tstr in _PROTECTED:
                prev_is_dot = False
                continue
            field = "attribute" if prev_is_dot else "variable"
            st = _pos_to_offset(line_starts, tok.start)
            ed = _pos_to_offset(line_starts, tok.end)
            occ[(field, tstr)].append((st, ed))
            prev_is_dot = False
        elif ttype == tokenize.STRING:
            kind = classify_string_kind(tstr, route_cfg)
            if kind == "A":
                st = _pos_to_offset(line_starts, tok.start)
                ed = _pos_to_offset(line_starts, tok.end)
                occ[("string", tstr)].append((st, ed))
            prev_is_dot = False
        elif ttype == tokenize.OP:
            prev_is_dot = tstr == "."
        else:
            prev_is_dot = False

    candidates = len(occ)
    alias_candidates = _choose_aliases(_build_alias_pool(max(1, candidates * 2)), tokenizer, tok_type)
    alias_iter = iter(alias_candidates)
    selected: dict[tuple[str, str], str] = {}
    entries: list[AEntry] = []
    for key, spans in sorted(occ.items(), key=lambda kv: len(kv[1]), reverse=True):
        field, literal = key
        count = len(spans)
        if count < 2:
            continue
        alias_base = next(alias_iter, None)
        if not alias_base:
            break
        alias_surface = alias_base if field != "string" else repr(alias_base)
        raw_cost = _token_len(tokenizer, tok_type, literal)
        alias_cost = _token_len(tokenizer, tok_type, alias_surface)
        intro_entry = {
            "token": alias_surface,
            "kind": "stage3_ab_a_alias",
            "field": field,
            "definition": literal,
        }
        intro_cost = compute_vocab_intro_cost(
            [intro_entry],
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
        gain = count * (raw_cost - alias_cost) - intro_cost
        if gain <= 0:
            continue
        selected[key] = alias_surface
        entries.append(
            AEntry(
                field=field,
                literal=literal,
                alias=alias_surface,
                count=count,
                raw_cost=raw_cost,
                alias_cost=alias_cost,
                intro_cost=intro_cost,
                gain=gain,
            )
        )

    spans_all: list[tuple[int, int, str]] = []
    for key, alias in selected.items():
        for st, ed in occ[key]:
            spans_all.append((st, ed, alias))
    encoded = _apply_spans(text, spans_all) if spans_all else text

    vocab_entries = [
        {
            "token": e.alias,
            "kind": "stage3_ab_a_alias",
            "field": e.field,
            "definition": e.literal,
        }
        for e in entries
    ]
    seq_saved = sum(e.count * max(0, e.raw_cost - e.alias_cost) for e in entries)
    intro = sum(e.intro_cost for e in entries)
    return ACodecResult(
        encoded_text=encoded,
        entries=entries,
        candidates=candidates,
        selected=len(entries),
        used_entries=len(entries),
        intro_tokens=intro,
        sequence_saved=seq_saved,
        effective_net_saving=seq_saved - intro,
        vocab_entries=vocab_entries,
    )


def decode_exact_aliases(text: str, entries: list[AEntry]) -> str:
    """Exact decode for A channel aliases."""
    if not entries:
        return text
    name_map: dict[str, str] = {}
    string_map: dict[str, str] = {}
    for e in entries:
        if e.field == "string":
            string_map[e.alias] = e.literal
        else:
            name_map[e.alias] = e.literal
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text
    line_starts = _line_start_offsets(text)
    spans: list[tuple[int, int, str]] = []
    for tok in toks:
        rep = None
        if tok.type == tokenize.NAME:
            rep = name_map.get(tok.string)
        elif tok.type == tokenize.STRING:
            rep = string_map.get(tok.string)
        if rep is not None:
            st = _pos_to_offset(line_starts, tok.start)
            ed = _pos_to_offset(line_starts, tok.end)
            spans.append((st, ed, rep))
    return _apply_spans(text, spans)
