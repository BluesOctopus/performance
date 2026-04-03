"""Request-local exact aliasing for Stage3 hybrid A channel."""

from __future__ import annotations

import ast
import builtins
import io
import json
import keyword
import re
import tokenize
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from config import CACHE_DIR, VOCAB_COST_MODE
from placeholder_accounting import compute_vocab_intro_cost
from token_scorer import _line_start_offsets, _pos_to_offset

from ..routing.router import ABRoutingConfig, classify_string_with_reason

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
    reject_reason_counts: dict[str, int] = field(default_factory=dict)
    protected_name_count: int = 0
    min_occ_reject_count: int = 0
    net_gain_reject_count: int = 0


def _token_len(tokenizer: Any, tok_type: str, text: str) -> int:
    from marker_count import encode as _encode

    return len(_encode(tokenizer, tok_type, text))


def _alias_cache_id(tokenizer: Any, tok_type: str) -> str:
    model_name = getattr(tokenizer, "name_or_path", "") or getattr(tokenizer, "model", "")
    if not isinstance(model_name, str):
        model_name = str(model_name)
    cls_name = f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    raw = f"{tok_type}_{model_name or cls_name}".strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
    return safe or f"{tok_type}_tokenizer"


def _load_alias_alphabet_cache(tokenizer: Any, tok_type: str) -> dict[str, Any]:
    cache_id = _alias_cache_id(tokenizer, tok_type)
    fp = CACHE_DIR / "alias_alphabets" / f"alias_alphabet_{cache_id}.json"
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_alias_alphabet_cache(tokenizer: Any, tok_type: str, payload: dict[str, Any]) -> None:
    cache_id = _alias_cache_id(tokenizer, tok_type)
    fp = CACHE_DIR / "alias_alphabets" / f"alias_alphabet_{cache_id}.json"
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def build_alias_alphabet(
    tokenizer: Any,
    tok_type: str,
    *,
    style: str,
    prefix: Optional[str] = None,
    max_n: int = 256,
    candidate_style: str = "token_cost_sorted",
) -> list[str]:
    style = (style or "").strip().lower()
    style = style if style in {"short", "mnemonic"} else "short"
    cache = _load_alias_alphabet_cache(tokenizer, tok_type)

    if style == "short":
        cached = cache.get("short", {})
        if (
            isinstance(cached, dict)
            and cached.get("max_n") == max_n
            and str(cached.get("candidate_style", "token_cost_sorted")) == candidate_style
        ):
            cands = cached.get("candidates")
            if isinstance(cands, list) and all(isinstance(x, str) for x in cands):
                return cands
        items: list[tuple[int, int, str]] = []
        families: list[str]
        if candidate_style == "compact_mixed":
            families = ["x", "v", "_x", "__ab"]
        elif candidate_style == "underscore_heavy":
            families = ["__ab", "__x", "_x", "x"]
        else:
            families = ["__ab", "_x", "x", "z"]
        per_family = max(8, max_n // max(1, len(families)))
        generated: list[str] = []
        for fam in families:
            for n in range(per_family):
                generated.append(f"{fam}{n}")
        # Ensure stable budget and uniqueness.
        seen_alias: set[str] = set()
        candidates = []
        for a in generated:
            if a in seen_alias:
                continue
            seen_alias.add(a)
            candidates.append(a)
            if len(candidates) >= max_n:
                break
        for alias in candidates:
            items.append((_token_len(tokenizer, tok_type, alias), len(alias), alias))
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        out = [a for _c, _l, a in items]
        cache["short"] = {
            "max_n": max_n,
            "candidate_style": candidate_style,
            "candidates": out,
        }
        _save_alias_alphabet_cache(tokenizer, tok_type, cache)
        return out

    if not prefix:
        prefix = "x"
    cached_mn = cache.get("mnemonic_prefixes", {})
    if isinstance(cached_mn, dict):
        entry = cached_mn.get(prefix, {})
        if isinstance(entry, dict) and entry.get("max_n") == max_n:
            cands = entry.get("candidates")
            if isinstance(cands, list) and all(isinstance(x, str) for x in cands):
                return cands
    items = []
    for n in range(max_n):
        alias = f"_{prefix}{n}"
        items.append((_token_len(tokenizer, tok_type, alias), len(alias), alias))
    items.sort(key=lambda t: (t[0], t[1], t[2]))
    out = [a for _c, _l, a in items]
    cache.setdefault("mnemonic_prefixes", {})[prefix] = {"max_n": max_n, "candidates": out}
    _save_alias_alphabet_cache(tokenizer, tok_type, cache)
    return out


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    out = text
    for st, ed, rep in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:st] + rep + out[ed:]
    return out


def _sanitize_prefix(name: str) -> str:
    s = "".join(ch for ch in name if ch.isalnum() or ch == "_")
    return (s[:2] if s else "x").lower()


def _collect_ast_protected_names(text: str) -> set[str]:
    out: set[str] = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError):
        return out
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            if re.match(r"^__.*__$", name):
                out.add(name)
                continue
            if name.startswith("_"):
                continue
            out.add(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out.add(elt.value)
    return out


def encode_exact_aliases(
    text: str,
    *,
    tokenizer: Any,
    tok_type: str,
    route_cfg: ABRoutingConfig,
    min_occ: int = 2,
    min_net_gain: int = 1,
    alias_style: str = "short",
    alias_candidate_style: str = "token_cost_sorted",
) -> ACodecResult:
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ACodecResult(encoded_text=text)
    line_starts = _line_start_offsets(text)
    ast_protected = _collect_ast_protected_names(text)
    occ: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    reject_reasons: Counter[str] = Counter()
    protected_name_count = 0
    prev_is_dot = False
    for tok in toks:
        ttype, tstr = tok.type, tok.string
        if ttype == tokenize.NAME:
            if tstr in _PROTECTED or tstr in ast_protected:
                protected_name_count += 1
                prev_is_dot = False
                continue
            field = "attribute" if prev_is_dot else "variable"
            st = _pos_to_offset(line_starts, tok.start)
            ed = _pos_to_offset(line_starts, tok.end)
            occ[(field, tstr)].append((st, ed))
            prev_is_dot = False
        elif ttype == tokenize.STRING:
            route, reason = classify_string_with_reason(tstr, route_cfg)
            if route == "A":
                st = _pos_to_offset(line_starts, tok.start)
                ed = _pos_to_offset(line_starts, tok.end)
                occ[("string", tstr)].append((st, ed))
            else:
                reject_reasons[f"route_{reason}"] += 1
            prev_is_dot = False
        elif ttype == tokenize.OP:
            prev_is_dot = tstr == "."
        else:
            prev_is_dot = False

    candidates = len(occ)
    alias_style = (alias_style or "").strip().lower()
    if alias_style not in {"short", "mnemonic"}:
        alias_style = "short"
    alias_iter_idx = 0
    short_alias_alphabet: list[str] = []
    mnemonic_alias_alphabet_by_prefix: dict[str, list[str]] = {}
    mnemonic_cursor_by_prefix: defaultdict[str, int] = defaultdict(int)
    if alias_style == "short":
        short_alias_alphabet = build_alias_alphabet(
            tokenizer,
            tok_type,
            style="short",
            max_n=256,
            candidate_style=alias_candidate_style,
        )

    selected: dict[tuple[str, str], str] = {}
    entries: list[AEntry] = []
    min_occ_reject_count = 0
    net_gain_reject_count = 0
    for key, spans in sorted(occ.items(), key=lambda kv: len(kv[1]), reverse=True):
        field, literal = key
        count = len(spans)
        if count < int(min_occ):
            min_occ_reject_count += 1
            continue
        if alias_style == "mnemonic":
            prefix = _sanitize_prefix(literal)
            if prefix not in mnemonic_alias_alphabet_by_prefix:
                mnemonic_alias_alphabet_by_prefix[prefix] = build_alias_alphabet(
                    tokenizer, tok_type, style="mnemonic", prefix=prefix, max_n=256
                )
            cursor = mnemonic_cursor_by_prefix[prefix]
            alias_base = (
                mnemonic_alias_alphabet_by_prefix[prefix][cursor]
                if cursor < len(mnemonic_alias_alphabet_by_prefix[prefix])
                else f"_{prefix}{cursor}"
            )
        else:
            cursor = alias_iter_idx
            alias_base = short_alias_alphabet[cursor] if cursor < len(short_alias_alphabet) else f"__ab{cursor}"

        alias_surface = alias_base if field != "string" else repr(alias_base)
        raw_cost = _token_len(tokenizer, tok_type, literal)
        alias_cost = _token_len(tokenizer, tok_type, alias_surface)
        intro_entry = {"token": alias_surface, "kind": "stage3_ab_a_alias", "field": field, "definition": literal}
        intro_cost = compute_vocab_intro_cost([intro_entry], mode=VOCAB_COST_MODE, tokenizer=tokenizer, tok_type=tok_type)
        gain = count * (raw_cost - alias_cost) - intro_cost
        if gain < int(min_net_gain):
            net_gain_reject_count += 1
            continue
        if alias_style == "mnemonic":
            mnemonic_cursor_by_prefix[prefix] += 1
        else:
            alias_iter_idx += 1
        selected[key] = alias_surface
        entries.append(AEntry(field=field, literal=literal, alias=alias_surface, count=count, raw_cost=raw_cost, alias_cost=alias_cost, intro_cost=intro_cost, gain=gain))

    spans_all: list[tuple[int, int, str]] = []
    for key, alias in selected.items():
        for st, ed in occ[key]:
            spans_all.append((st, ed, alias))
    encoded = _apply_spans(text, spans_all) if spans_all else text

    vocab_entries = [{"token": e.alias, "kind": "stage3_ab_a_alias", "field": e.field, "definition": e.literal} for e in entries]
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
        reject_reason_counts=dict(reject_reasons),
        protected_name_count=protected_name_count,
        min_occ_reject_count=min_occ_reject_count,
        net_gain_reject_count=net_gain_reject_count,
    )


def decode_exact_aliases(text: str, entries: list[AEntry]) -> str:
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

