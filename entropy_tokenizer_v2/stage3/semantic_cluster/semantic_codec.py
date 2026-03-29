"""Lightweight semantic clustering codec for Stage3 hybrid B channel."""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from placeholder_accounting import compute_vocab_intro_cost
from token_scorer import _line_start_offsets, _pos_to_offset

from semantic_cluster.string_classifier import (
    SemanticClassifierConfig,
    is_semantic_free_text,
)
from config import VOCAB_COST_MODE

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


@dataclass(slots=True)
class BCluster:
    code: str
    representative: str
    members: list[str]
    avg_similarity: float


@dataclass(slots=True)
class BCodecResult:
    encoded_text: str
    clusters: list[BCluster] = field(default_factory=list)
    candidates: int = 0
    cluster_count: int = 0
    used_clusters: int = 0
    intro_tokens: int = 0
    sequence_saved: int = 0
    effective_net_saving: int = 0
    fallback_count: int = 0
    risk_reject_count: int = 0
    avg_similarity: float = 0.0
    vocab_entries: list[dict[str, Any]] = field(default_factory=list)


def _token_len(tokenizer: Any, tok_type: str, text: str) -> int:
    from marker_count import encode as _encode

    return len(_encode(tokenizer, tok_type, text))


def _vec(text: str) -> Counter[str]:
    return Counter(x.lower() for x in _WORD_RE.findall(text))


def _cos(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    out = text
    for st, ed, rep in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:st] + rep + out[ed:]
    return out


def encode_semantic_strings(
    text: str,
    *,
    tokenizer: Any,
    tok_type: str,
    similarity_threshold: float = 0.82,
    risk_threshold: float = 0.72,
    min_cluster_size: int = 2,
    classifier_cfg: SemanticClassifierConfig | None = None,
    code_prefix: str = "__abB",
) -> BCodecResult:
    cfg = classifier_cfg or SemanticClassifierConfig()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return BCodecResult(encoded_text=text)

    line_starts = _line_start_offsets(text)
    occurrences: dict[str, list[tuple[int, int]]] = {}
    inners: dict[str, str] = {}
    for tok in toks:
        if tok.type != tokenize.STRING:
            continue
        sp = tok.string
        if not is_semantic_free_text(sp, cfg):
            continue
        try:
            inner = ast.literal_eval(sp)
        except (SyntaxError, ValueError, MemoryError):
            continue
        if not isinstance(inner, str):
            continue
        st = _pos_to_offset(line_starts, tok.start)
        ed = _pos_to_offset(line_starts, tok.end)
        occurrences.setdefault(sp, []).append((st, ed))
        inners[sp] = inner

    cands = len(occurrences)
    if cands == 0:
        return BCodecResult(encoded_text=text)

    vecs = {sp: _vec(inner) for sp, inner in inners.items()}
    literals = list(occurrences.keys())
    clusters: list[list[str]] = []
    for sp in literals:
        placed = False
        for cl in clusters:
            sims = [_cos(vecs[sp], vecs[other]) for other in cl]
            if sims and sum(sims) / len(sims) >= similarity_threshold:
                cl.append(sp)
                placed = True
                break
        if not placed:
            clusters.append([sp])

    usable: list[BCluster] = []
    replacements: dict[str, str] = {}
    vocab_entries: list[dict[str, Any]] = []
    risk_reject_count = 0
    fallback_count = 0
    seq_saved = 0
    intro = 0
    sim_values: list[float] = []

    for idx, members in enumerate(clusters):
        if len(members) < min_cluster_size:
            fallback_count += len(members)
            continue
        rep = min(members, key=lambda s: len(inners[s]))
        rep_vec = vecs[rep]
        sims = [_cos(vecs[m], rep_vec) for m in members if m != rep]
        avg_sim = sum(sims) / len(sims) if sims else 1.0
        if avg_sim < risk_threshold:
            risk_reject_count += len(members)
            continue
        code_inner = f"{code_prefix}{idx}"
        code_surface = repr(code_inner)
        intro_entry = {
            "token": code_surface,
            "kind": "stage3_ab_b_cluster",
            "definition": inners[rep],
            "cluster_size": len(members),
        }
        intro_cost = compute_vocab_intro_cost(
            [intro_entry],
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
        raw_total = 0
        code_total = 0
        for m in members:
            cnt = len(occurrences[m])
            raw_total += cnt * _token_len(tokenizer, tok_type, m)
            code_total += cnt * _token_len(tokenizer, tok_type, code_surface)
        if raw_total - code_total <= intro_cost:
            fallback_count += len(members)
            continue
        for m in members:
            replacements[m] = code_surface
        usable.append(
            BCluster(
                code=code_inner,
                representative=inners[rep],
                members=list(members),
                avg_similarity=avg_sim,
            )
        )
        vocab_entries.append(intro_entry)
        seq_saved += raw_total - code_total
        intro += intro_cost
        sim_values.append(avg_sim)

    spans: list[tuple[int, int, str]] = []
    for lit, rep in replacements.items():
        for st, ed in occurrences.get(lit, []):
            spans.append((st, ed, rep))
    encoded = _apply_spans(text, spans) if spans else text

    return BCodecResult(
        encoded_text=encoded,
        clusters=usable,
        candidates=cands,
        cluster_count=len(clusters),
        used_clusters=len(usable),
        intro_tokens=intro,
        sequence_saved=seq_saved,
        effective_net_saving=seq_saved - intro,
        fallback_count=fallback_count,
        risk_reject_count=risk_reject_count,
        avg_similarity=(sum(sim_values) / len(sim_values)) if sim_values else 0.0,
        vocab_entries=vocab_entries,
    )
