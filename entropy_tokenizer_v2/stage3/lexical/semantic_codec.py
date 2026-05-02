"""Lightweight lexical clustering codec for Stage3 hybrid B channel."""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from config import VOCAB_COST_MODE
from placeholder_accounting import compute_vocab_intro_cost
from telemetry_summary import int_summary
from token_scorer import _line_start_offsets, _pos_to_offset

from .string_classifier import SemanticClassifierConfig, classify_semantic_free_text

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
    similarity_kind: str = "lexical_bow_cosine"
    mode: str = "lexical_free_text_baseline"
    reject_reason_counts: dict[str, int] = field(default_factory=dict)
    intro_not_worth_count: int = 0
    # Funnel / cluster-level telemetry (Phase 0 hybrid_ab).
    b_free_text_candidates_total: int = 0
    b_free_text_candidates_visible_after_stage2: int = 0
    b_clusters_formed: int = 0
    b_clusters_rejected_too_small: int = 0
    b_clusters_rejected_similarity_or_quality: int = 0
    b_clusters_rejected_intro_cost: int = 0
    b_clusters_selected_final: int = 0
    b_raw_total_summary: dict[str, float | int] = field(default_factory=dict)
    b_code_total_summary: dict[str, float | int] = field(default_factory=dict)
    b_intro_cost_summary: dict[str, float | int] = field(default_factory=dict)


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


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    if uni <= 0:
        return 0.0
    return inter / uni


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
    similarity_kind: str = "lexical_bow_cosine",
    lexical_weight: float = 0.7,
    char_weight: float = 0.3,
    ngram_n: int = 3,
) -> BCodecResult:
    cfg = classifier_cfg or SemanticClassifierConfig()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return BCodecResult(encoded_text=text)

    line_starts = _line_start_offsets(text)
    occurrences: dict[str, list[tuple[int, int]]] = {}
    inners: dict[str, str] = {}
    route_rejects: Counter[str] = Counter()
    string_literal_candidates_total = 0
    for tok in toks:
        if tok.type != tokenize.STRING:
            continue
        string_literal_candidates_total += 1
        sp = tok.string
        route, reason = classify_semantic_free_text(sp, cfg)
        if route != "B":
            route_rejects[f"route_{reason}"] += 1
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
        return BCodecResult(
            encoded_text=text,
            b_free_text_candidates_total=string_literal_candidates_total,
            b_free_text_candidates_visible_after_stage2=0,
        )

    vecs = {sp: _vec(inner) for sp, inner in inners.items()}
    grams = {sp: _char_ngrams(inner, n=max(2, int(ngram_n))) for sp, inner in inners.items()}
    use_mixed = (similarity_kind or "").strip().lower() in {"hybrid_lexical_char", "mixed"}
    lw = max(0.0, float(lexical_weight))
    cw = max(0.0, float(char_weight))
    wsum = max(1e-9, lw + cw)

    def _sim(lhs: str, rhs: str) -> float:
        lexical = _cos(vecs[lhs], vecs[rhs])
        if not use_mixed:
            return lexical
        char = _jaccard(grams[lhs], grams[rhs])
        return (lw * lexical + cw * char) / wsum

    literals = list(occurrences.keys())
    clusters: list[list[str]] = []
    for sp in literals:
        placed = False
        for cl in clusters:
            sims = [_sim(sp, other) for other in cl]
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
    intro_not_worth_count = 0
    seq_saved = 0
    intro = 0
    sim_values: list[float] = []
    clusters_too_small = 0
    clusters_sim_reject = 0
    clusters_intro_reject = 0
    raw_totals_seen: list[int] = []
    code_totals_seen: list[int] = []
    intro_costs_seen: list[int] = []

    for idx, members in enumerate(clusters):
        if len(members) < min_cluster_size:
            clusters_too_small += 1
            fallback_count += len(members)
            continue
        rep = min(
            members,
            key=lambda cand: (
                _token_len(tokenizer, tok_type, cand),
                -(
                    sum(_sim(cand, other) for other in members if other != cand)
                    / max(1, len(members) - 1)
                ),
                len(inners[cand]),
            ),
        )
        sims = [_sim(m, rep) for m in members if m != rep]
        avg_sim = sum(sims) / len(sims) if sims else 1.0
        if avg_sim < risk_threshold:
            clusters_sim_reject += 1
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
        raw_totals_seen.append(raw_total)
        code_totals_seen.append(code_total)
        intro_costs_seen.append(int(intro_cost))
        if raw_total - code_total <= intro_cost:
            clusters_intro_reject += 1
            intro_not_worth_count += 1
            fallback_count += len(members)
            continue
        for m in members:
            replacements[m] = code_surface
        usable.append(BCluster(code=code_inner, representative=inners[rep], members=list(members), avg_similarity=avg_sim))
        vocab_entries.append(intro_entry)
        seq_saved += raw_total - code_total
        intro += intro_cost
        sim_values.append(avg_sim)

    spans: list[tuple[int, int, str]] = []
    for lit, rep in replacements.items():
        for st, ed in occurrences.get(lit, []):
            spans.append((st, ed, rep))
    encoded = _apply_spans(text, spans) if spans else text

    sel_raw: list[int] = []
    sel_code: list[int] = []
    sel_intro: list[int] = []
    for cl in usable:
        code_surface = repr(cl.code)
        rt = sum(len(occurrences[m]) * _token_len(tokenizer, tok_type, m) for m in cl.members)
        ct = sum(len(occurrences[m]) * _token_len(tokenizer, tok_type, code_surface) for m in cl.members)
        ie = {
            "token": code_surface,
            "kind": "stage3_ab_b_cluster",
            "definition": cl.representative,
            "cluster_size": len(cl.members),
        }
        ic = int(
            compute_vocab_intro_cost(
                [ie], mode=VOCAB_COST_MODE, tokenizer=tokenizer, tok_type=tok_type
            )
        )
        sel_raw.append(rt)
        sel_code.append(ct)
        sel_intro.append(ic)

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
        similarity_kind="hybrid_lexical_char" if use_mixed else "lexical_bow_cosine",
        mode="lexical_free_text_mixed" if use_mixed else "lexical_free_text_baseline",
        reject_reason_counts=dict(route_rejects),
        intro_not_worth_count=intro_not_worth_count,
        b_free_text_candidates_total=string_literal_candidates_total,
        b_free_text_candidates_visible_after_stage2=cands,
        b_clusters_formed=len(clusters),
        b_clusters_rejected_too_small=clusters_too_small,
        b_clusters_rejected_similarity_or_quality=clusters_sim_reject,
        b_clusters_rejected_intro_cost=clusters_intro_reject,
        b_clusters_selected_final=len(usable),
        b_raw_total_summary=int_summary(sel_raw if sel_raw else raw_totals_seen),
        b_code_total_summary=int_summary(sel_code if sel_code else code_totals_seen),
        b_intro_cost_summary=int_summary(sel_intro if sel_intro else intro_costs_seen, with_min=True),
    )

