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
from telemetry_summary import int_summary
from token_scorer import _line_start_offsets, _pos_to_offset, sum_context_aware_literal_delta

from ..literal_codec.alias_pool import build_legal_alias_alphabet
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
    rejected_raw_too_short: int = 0
    rejected_alias_conflict: int = 0
    rejected_gain_non_positive: int = 0
    rejected_context_rescore_negative: int = 0
    a_raw_token_len_summary: dict[str, float | int] = field(default_factory=dict)
    a_alias_token_len_summary: dict[str, float | int] = field(default_factory=dict)
    a_context_delta_summary: dict[str, float | int] = field(default_factory=dict)
    # Span index for incremental guardrail rollback (hybrid_ab file-level check).
    occ: dict[tuple[str, str], list[tuple[int, int]]] = field(default_factory=dict)


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


def _collect_scope_name_conflicts(text: str) -> set[str]:
    """Names / attributes / imports that must not be reused as A-channel aliases."""
    names: set[str] = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError):
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                base = (a.name or "").split(".")[0]
                names.add(a.asname or base)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                names.add(a.asname or a.name)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.NamedExpr):
            names.add(node.target.id)
    return names


def apply_a_entries(
    text: str,
    occ: dict[tuple[str, str], list[tuple[int, int]]],
    entries: list[AEntry],
) -> str:
    """Re-apply a subset of A entries onto Stage2 text (used by file-level guardrail)."""
    if not entries:
        return text
    spans_all: list[tuple[int, int, str]] = []
    for e in entries:
        key = (e.field, e.literal)
        for st, ed in occ.get(key, ()):
            spans_all.append((st, ed, e.alias))
    return _apply_spans(text, spans_all) if spans_all else text


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
    cost_mode: str = "local",
    min_raw_token_len: int = 1,
    max_alias_token_len: int = 32,
    context_window_chars: int = 80,
    a_alias_rank_pool_cap: int = 32,
    a_context_gain_margin: int = 0,
) -> ACodecResult:
    cost_mode = (cost_mode or "local").strip().lower()
    if cost_mode not in {"local", "context_aware"}:
        cost_mode = "local"
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ACodecResult(encoded_text=text)
    line_starts = _line_start_offsets(text)
    ast_protected = _collect_ast_protected_names(text)
    scope_conflicts = _collect_scope_name_conflicts(text)
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
    alias_candidate_style = (alias_candidate_style or "token_cost_sorted").strip().lower()
    alias_iter_idx = 0
    short_alias_alphabet: list[str] = []
    legal_alias_idx = 0
    mnemonic_alias_alphabet_by_prefix: dict[str, list[str]] = {}
    mnemonic_cursor_by_prefix: defaultdict[str, int] = defaultdict(int)
    taken_alias_bases: set[str] = set()
    static_reserved = set(_PROTECTED) | ast_protected | scope_conflicts

    if alias_style == "short":
        if alias_candidate_style == "legal_identifier_pool":
            short_alias_alphabet = build_legal_alias_alphabet(
                tokenizer,
                tok_type,
                reserved=static_reserved,
                max_n=256,
                max_alias_token_len=int(max_alias_token_len),
            )
        else:
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
    rejected_raw_too_short = 0
    rejected_alias_conflict = 0
    rejected_gain_non_positive = 0
    rejected_context_rescore_negative = 0
    sel_raw_lens: list[int] = []
    sel_alias_lens: list[int] = []
    sel_ctx_deltas: list[int] = []
    rank_cap = max(1, int(a_alias_rank_pool_cap))
    gain_margin = int(a_context_gain_margin)
    min_ng = int(min_net_gain)

    for key, spans in sorted(occ.items(), key=lambda kv: len(kv[1]), reverse=True):
        field, literal = key
        count = len(spans)
        if count < int(min_occ):
            min_occ_reject_count += 1
            continue

        pool: list[tuple[str, str]] = []

        if alias_style == "mnemonic":
            prefix = _sanitize_prefix(literal)
            if prefix not in mnemonic_alias_alphabet_by_prefix:
                mnemonic_alias_alphabet_by_prefix[prefix] = build_alias_alphabet(
                    tokenizer, tok_type, style="mnemonic", prefix=prefix, max_n=256
                )
            c = mnemonic_cursor_by_prefix[prefix]
            attempts = 0
            while len(pool) < rank_cap and attempts < 512:
                if c < len(mnemonic_alias_alphabet_by_prefix[prefix]):
                    ab = mnemonic_alias_alphabet_by_prefix[prefix][c]
                else:
                    ab = f"_{prefix}{c}"
                c += 1
                attempts += 1
                surf = ab if field != "string" else repr(ab)
                if ab in taken_alias_bases or ab in static_reserved:
                    continue
                if _token_len(tokenizer, tok_type, surf) > int(max_alias_token_len):
                    continue
                pool.append((ab, surf))
            mnemonic_cursor_by_prefix[prefix] = c
        elif alias_candidate_style == "legal_identifier_pool":
            i = legal_alias_idx
            while len(pool) < rank_cap and i < len(short_alias_alphabet):
                ab = short_alias_alphabet[i]
                i += 1
                if ab in taken_alias_bases or ab in static_reserved:
                    continue
                surf = ab if field != "string" else repr(ab)
                if _token_len(tokenizer, tok_type, surf) > int(max_alias_token_len):
                    continue
                pool.append((ab, surf))
            legal_alias_idx = i
        else:
            attempt = 0
            start_iter = alias_iter_idx
            while len(pool) < rank_cap and attempt < 512:
                cursor = start_iter + attempt
                if cursor < len(short_alias_alphabet):
                    ab = short_alias_alphabet[cursor]
                else:
                    ab = f"x{cursor}"
                attempt += 1
                surf = ab if field != "string" else repr(ab)
                if ab in taken_alias_bases or ab in static_reserved:
                    continue
                if _token_len(tokenizer, tok_type, surf) > int(max_alias_token_len):
                    continue
                pool.append((ab, surf))
            alias_iter_idx = start_iter + attempt

        if not pool:
            rejected_alias_conflict += 1
            net_gain_reject_count += 1
            continue

        raw_cost = _token_len(tokenizer, tok_type, literal)
        if raw_cost < int(min_raw_token_len):
            rejected_raw_too_short += 1
            net_gain_reject_count += 1
            continue

        scored: list[tuple[str, str, int, int, int, int]] = []
        for ab, surf in pool:
            alias_cost = _token_len(tokenizer, tok_type, surf)
            intro_entry = {"token": surf, "kind": "stage3_ab_a_alias", "field": field, "definition": literal}
            intro_cost = int(
                compute_vocab_intro_cost(
                    [intro_entry], mode=VOCAB_COST_MODE, tokenizer=tokenizer, tok_type=tok_type
                )
            )
            local_gain = count * (raw_cost - alias_cost) - intro_cost
            seq_delta = 0
            if cost_mode == "context_aware":
                seq_delta = int(
                    sum_context_aware_literal_delta(
                        text,
                        spans,
                        literal,
                        surf,
                        tokenizer,
                        tok_type,
                        window_chars=int(context_window_chars),
                    )
                )
            scored.append((ab, surf, intro_cost, alias_cost, local_gain, seq_delta))

        qualified = [t for t in scored if t[4] >= min_ng]
        if not qualified:
            rejected_gain_non_positive += 1
            net_gain_reject_count += 1
            continue

        def _accept_net(t: tuple[str, str, int, int, int, int]) -> int:
            if cost_mode == "context_aware":
                return int(t[5]) - int(t[2])
            return int(t[4])

        best = max(qualified, key=_accept_net)
        accept_net = _accept_net(best)
        if accept_net < min_ng + gain_margin:
            if cost_mode == "context_aware":
                rejected_context_rescore_negative += 1
            else:
                rejected_gain_non_positive += 1
            net_gain_reject_count += 1
            continue

        alias_base, alias_surface = best[0], best[1]
        intro_cost, alias_cost = best[2], best[3]
        gain = accept_net
        taken_alias_bases.add(alias_base)

        selected[key] = alias_surface
        seq_d = int(best[5]) if cost_mode == "context_aware" else count * (raw_cost - alias_cost)
        sel_raw_lens.append(raw_cost)
        sel_alias_lens.append(alias_cost)
        sel_ctx_deltas.append(seq_d)
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

    vocab_entries = [{"token": e.alias, "kind": "stage3_ab_a_alias", "field": e.field, "definition": e.literal} for e in entries]
    seq_saved = sum(e.count * max(0, e.raw_cost - e.alias_cost) for e in entries)
    intro = sum(e.intro_cost for e in entries)
    occ_snapshot = {k: list(v) for k, v in occ.items()}
    ctx_sum = int_summary(sel_ctx_deltas, with_min=True)
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
        rejected_raw_too_short=rejected_raw_too_short,
        rejected_alias_conflict=rejected_alias_conflict,
        rejected_gain_non_positive=rejected_gain_non_positive,
        rejected_context_rescore_negative=rejected_context_rescore_negative,
        a_raw_token_len_summary=int_summary(sel_raw_lens),
        a_alias_token_len_summary=int_summary(sel_alias_lens),
        a_context_delta_summary=ctx_sum,
        occ=occ_snapshot,
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

