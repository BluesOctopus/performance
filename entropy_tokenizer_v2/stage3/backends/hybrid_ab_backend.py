from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from config import VOCAB_COST_MODE
from marker_count import encode as mc_encode
from placeholder_accounting import compute_vocab_intro_cost
from stage3.backends.base import Stage3EncodeResult
from stage3.exact.alias_codec import ACodecResult, AEntry, apply_a_entries, encode_exact_aliases
from stage3.lexical.semantic_codec import BCodecResult, encode_semantic_strings
from stage3.lexical.string_classifier import SemanticClassifierConfig
from stage3.routing.router import ABRoutingConfig


@dataclass(slots=True)
class HybridABConfig:
    mode: str = "exact_only"
    free_text_min_chars: int = 24
    free_text_min_words: int = 4
    key_like_patterns: tuple[str, ...] = ()
    short_string_policy: str = "exact_candidate"
    enable_mid_free_text: bool = False
    free_text_mid_min_chars: int = 14
    free_text_mid_min_words: int = 3
    allow_multiline_whitelist: bool = False
    multiline_max_lines: int = 3
    multiline_max_chars: int = 220
    a_min_occ: int = 2
    a_min_net_gain: int = 1
    a_alias_style: str = "short"
    a_alias_candidate_style: str = "token_cost_sorted"
    b_similarity_threshold: float = 0.82
    b_risk_threshold: float = 0.72
    b_min_cluster_size: int = 2
    b_similarity_kind: str = "lexical_bow_cosine"
    b_lexical_weight: float = 0.7
    b_char_weight: float = 0.3
    b_char_ngram_n: int = 3
    a_processing_mode: str = "full"
    a_cost_mode: str = "local"
    enable_global_guardrail: bool = False
    enable_incremental_rollback: bool = False
    min_raw_token_len: int = 1
    max_alias_token_len: int = 32
    context_window_chars: int = 80
    b_channel_priority: str = "normal"
    a_alias_rank_pool_cap: int = 32
    a_context_gain_margin: int = 0
    a_enable_local_combo_greedy: bool = False
    a_combo_max_additions: int = 24


@dataclass(slots=True)
class HybridABResult:
    encoded_text: str
    a: ACodecResult
    b: BCodecResult
    fallback_count: int
    meta: dict[str, Any] = field(default_factory=dict)


def _ekey(e: AEntry) -> tuple[str, str]:
    return (e.field, e.literal)


def _sequence_token_len(text: str, tokenizer: Any, tok_type: str) -> int:
    return len(mc_encode(tokenizer, tok_type, text))


def _encode_b_channel(
    text_a: str,
    *,
    conf: HybridABConfig,
    tokenizer: Any,
    tok_type: str,
) -> BCodecResult:
    if conf.mode != "hybrid":
        return BCodecResult(encoded_text=text_a)
    return encode_semantic_strings(
        text_a,
        tokenizer=tokenizer,
        tok_type=tok_type,
        similarity_threshold=conf.b_similarity_threshold,
        risk_threshold=conf.b_risk_threshold,
        min_cluster_size=conf.b_min_cluster_size,
        classifier_cfg=SemanticClassifierConfig(
            free_text_min_chars=conf.free_text_min_chars,
            free_text_min_words=conf.free_text_min_words,
            enable_mid_free_text=conf.enable_mid_free_text,
            free_text_mid_min_chars=conf.free_text_mid_min_chars,
            free_text_mid_min_words=conf.free_text_mid_min_words,
            allow_multiline_whitelist=conf.allow_multiline_whitelist,
            multiline_max_lines=conf.multiline_max_lines,
            multiline_max_chars=conf.multiline_max_chars,
        ),
        similarity_kind=conf.b_similarity_kind,
        lexical_weight=conf.b_lexical_weight,
        char_weight=conf.b_char_weight,
        ngram_n=conf.b_char_ngram_n,
    )


def _greedy_a_combo(
    text: str,
    occ: dict[tuple[str, str], list[tuple[int, int]]],
    entries: list[AEntry],
    *,
    tokenizer: Any,
    tok_type: str,
    max_add: int,
) -> list[AEntry]:
    """Incremental greedy subset using full-file sequence length (non-overlapping spans)."""
    if not entries:
        return []
    cap = max(1, int(max_add))
    sorted_e = sorted(entries, key=lambda e: -e.gain)[:cap]
    chosen: list[AEntry] = []
    for cand in sorted_e:
        trial = chosen + [cand]
        cur_text = apply_a_entries(text, occ, chosen)
        next_text = apply_a_entries(text, occ, trial)
        if _sequence_token_len(next_text, tokenizer, tok_type) < _sequence_token_len(
            cur_text, tokenizer, tok_type
        ):
            chosen.append(cand)
    return chosen


def _build_a_codec_subset(
    text_after_a: str,
    occ: dict[tuple[str, str], list[tuple[int, int]]],
    entries: list[AEntry],
    orig_a: ACodecResult,
) -> ACodecResult:
    intro = sum(e.intro_cost for e in entries)
    seq_saved = sum(e.count * max(0, e.raw_cost - e.alias_cost) for e in entries)
    vocab_entries = [
        {"token": e.alias, "kind": "stage3_ab_a_alias", "field": e.field, "definition": e.literal}
        for e in entries
    ]
    return ACodecResult(
        encoded_text=text_after_a,
        entries=list(entries),
        candidates=orig_a.candidates,
        selected=len(entries),
        used_entries=len(entries),
        intro_tokens=intro,
        sequence_saved=seq_saved,
        effective_net_saving=seq_saved - intro,
        vocab_entries=vocab_entries,
        reject_reason_counts=dict(orig_a.reject_reason_counts),
        protected_name_count=orig_a.protected_name_count,
        min_occ_reject_count=orig_a.min_occ_reject_count,
        net_gain_reject_count=orig_a.net_gain_reject_count,
        rejected_raw_too_short=orig_a.rejected_raw_too_short,
        rejected_alias_conflict=orig_a.rejected_alias_conflict,
        rejected_gain_non_positive=orig_a.rejected_gain_non_positive,
        rejected_context_rescore_negative=orig_a.rejected_context_rescore_negative,
        a_raw_token_len_summary=dict(orig_a.a_raw_token_len_summary),
        a_alias_token_len_summary=dict(orig_a.a_alias_token_len_summary),
        a_context_delta_summary=dict(orig_a.a_context_delta_summary),
        occ=dict(occ),
    )


def _rebuild_hybrid_from_entries(
    s2_text: str,
    occ: dict[tuple[str, str], list[tuple[int, int]]],
    entries: list[AEntry],
    orig_a: ACodecResult,
    conf: HybridABConfig,
    tokenizer: Any,
    tok_type: str,
) -> HybridABResult:
    text_a = apply_a_entries(s2_text, occ, entries)
    a_res = _build_a_codec_subset(text_a, occ, entries, orig_a)
    b_res = _encode_b_channel(text_a, conf=conf, tokenizer=tokenizer, tok_type=tok_type)
    return HybridABResult(
        encoded_text=b_res.encoded_text,
        a=a_res,
        b=b_res,
        fallback_count=b_res.fallback_count,
        meta={},
    )


def _apply_hybrid_ab_file_guardrail(
    s2_text: str,
    raw: HybridABResult,
    *,
    conf: HybridABConfig,
    tokenizer: Any,
    tok_type: str,
) -> tuple[HybridABResult, dict[str, Any]]:
    """
    Phase-1 safety: realized sequence tokens must not exceed Stage2.

    Phase-2: greedy A-entry rollback (largest isolated alias inflation first), then
    full Stage2 fallback if B/A interactions still inflate the file.
    """
    t2 = _sequence_token_len(s2_text, tokenizer, tok_type)
    t3 = _sequence_token_len(raw.encoded_text, tokenizer, tok_type)
    orig_entries = list(raw.a.entries)
    occ = dict(raw.a.occ or {})
    telem: dict[str, Any] = {
        "stage2_tokens": t2,
        "stage3_tokens": t3,
        "stage3_realized_delta": t2 - t3,
        "stage3_guardrail_triggered": False,
        "num_candidates_considered": raw.a.candidates,
        "num_candidates_applied": raw.a.selected,
        "num_aliases_rolled_back": 0,
    }
    if not conf.enable_global_guardrail:
        return raw, telem
    if t3 <= t2:
        return raw, telem

    telem["stage3_guardrail_triggered"] = True
    if not conf.enable_incremental_rollback or not orig_entries:
        empty_a = ACodecResult(encoded_text=s2_text, occ=occ)
        empty_b = BCodecResult(encoded_text=s2_text)
        telem.update(
            {
                "stage3_tokens": t2,
                "stage3_realized_delta": 0,
                "num_candidates_applied": 0,
                "num_aliases_rolled_back": len(orig_entries),
            }
        )
        return HybridABResult(s2_text, empty_a, empty_b, 0, {}), telem

    by_key = {_ekey(e): e for e in orig_entries}
    harm_queue = sorted(
        orig_entries,
        key=lambda e: -(e.count * max(0, e.alias_cost - e.raw_cost)),
    )
    remaining_keys = set(by_key.keys())
    best_snapshot = raw
    best_t = t3

    while remaining_keys:
        cur_entries = [by_key[k] for k in remaining_keys]
        rebuilt = _rebuild_hybrid_from_entries(
            s2_text, occ, cur_entries, raw.a, conf, tokenizer, tok_type
        )
        tn = _sequence_token_len(rebuilt.encoded_text, tokenizer, tok_type)
        if tn < best_t:
            best_snapshot, best_t = rebuilt, tn
        if tn <= t2:
            telem.update(
                {
                    "stage3_tokens": tn,
                    "stage3_realized_delta": t2 - tn,
                    "num_candidates_applied": len(cur_entries),
                    "num_aliases_rolled_back": len(orig_entries) - len(cur_entries),
                }
            )
            return rebuilt, telem
        victim_key = None
        while harm_queue:
            h = harm_queue.pop(0)
            hk = _ekey(h)
            if hk in remaining_keys:
                victim_key = hk
                break
        if victim_key is None:
            break
        remaining_keys.discard(victim_key)

    if best_t <= t2:
        cur_entries = list(best_snapshot.a.entries)
        telem.update(
            {
                "stage3_tokens": best_t,
                "stage3_realized_delta": t2 - best_t,
                "num_candidates_applied": len(cur_entries),
                "num_aliases_rolled_back": len(orig_entries) - len(cur_entries),
            }
        )
        return best_snapshot, telem

    empty_a = ACodecResult(encoded_text=s2_text, occ=occ)
    empty_b = BCodecResult(encoded_text=s2_text)
    telem.update(
        {
            "stage3_tokens": t2,
            "stage3_realized_delta": 0,
            "num_candidates_applied": 0,
            "num_aliases_rolled_back": len(orig_entries),
        }
    )
    return HybridABResult(s2_text, empty_a, empty_b, 0, {}), telem


def encode_stage3_hybrid_ab(
    text: str,
    *,
    tokenizer: Any,
    tok_type: str,
    cfg: HybridABConfig | None = None,
) -> HybridABResult:
    conf = cfg or HybridABConfig()
    route_cfg = ABRoutingConfig(
        free_text_min_chars=conf.free_text_min_chars,
        free_text_min_words=conf.free_text_min_words,
        fallback_unknown=True,
        key_like_patterns=conf.key_like_patterns,
        short_string_policy=conf.short_string_policy,
        enable_mid_free_text=conf.enable_mid_free_text,
        free_text_mid_min_chars=conf.free_text_mid_min_chars,
        free_text_mid_min_words=conf.free_text_mid_min_words,
        allow_multiline_whitelist=conf.allow_multiline_whitelist,
        multiline_max_lines=conf.multiline_max_lines,
        multiline_max_chars=conf.multiline_max_chars,
    )
    a_res = encode_exact_aliases(
        text,
        tokenizer=tokenizer,
        tok_type=tok_type,
        route_cfg=route_cfg,
        min_occ=conf.a_min_occ,
        min_net_gain=conf.a_min_net_gain,
        alias_style=conf.a_alias_style,
        alias_candidate_style=conf.a_alias_candidate_style,
        cost_mode=conf.a_cost_mode,
        min_raw_token_len=conf.min_raw_token_len,
        max_alias_token_len=conf.max_alias_token_len,
        context_window_chars=conf.context_window_chars,
        a_alias_rank_pool_cap=conf.a_alias_rank_pool_cap,
        a_context_gain_margin=conf.a_context_gain_margin,
    )
    if conf.a_enable_local_combo_greedy and a_res.entries:
        orig_a = a_res
        combo = _greedy_a_combo(
            text,
            a_res.occ,
            list(a_res.entries),
            tokenizer=tokenizer,
            tok_type=tok_type,
            max_add=conf.a_combo_max_additions,
        )
        if len(combo) != len(a_res.entries):
            text_a = apply_a_entries(text, a_res.occ, combo)
            a_res = _build_a_codec_subset(text_a, a_res.occ, combo, orig_a)
    b_res = _encode_b_channel(a_res.encoded_text, conf=conf, tokenizer=tokenizer, tok_type=tok_type)
    raw = HybridABResult(
        encoded_text=b_res.encoded_text,
        a=a_res,
        b=b_res,
        fallback_count=b_res.fallback_count,
        meta={},
    )
    final, guard_telem = _apply_hybrid_ab_file_guardrail(
        text, raw, conf=conf, tokenizer=tokenizer, tok_type=tok_type
    )
    a_res, b_res = final.a, final.b
    a_v = sum(1 for e in a_res.entries if e.field == "variable")
    a_a = sum(1 for e in a_res.entries if e.field == "attribute")
    a_s = sum(1 for e in a_res.entries if e.field == "string")
    meta = {
        "stage3_ab_a_candidates": a_res.candidates,
        "stage3_ab_a_selected": a_res.selected,
        "stage3_ab_a_used_entries": a_res.used_entries,
        "stage3_ab_a_used_entries_variable": a_v,
        "stage3_ab_a_used_entries_attribute": a_a,
        "stage3_ab_a_used_entries_string": a_s,
        "stage3_ab_a_intro_tokens": a_res.intro_tokens,
        "stage3_ab_a_sequence_saved": a_res.sequence_saved,
        "stage3_ab_a_effective_net_saving": a_res.effective_net_saving,
        "stage3_ab_a_protected_name_count": a_res.protected_name_count,
        "stage3_ab_a_min_occ_reject_count": a_res.min_occ_reject_count,
        "stage3_ab_a_net_gain_reject_count": a_res.net_gain_reject_count,
        "stage3_ab_a_reject_reason_counts": dict(a_res.reject_reason_counts),
        "a_candidates_total": a_res.candidates,
        "a_candidates_rejected_min_occ": a_res.min_occ_reject_count,
        "a_candidates_rejected_raw_too_short": a_res.rejected_raw_too_short,
        "a_candidates_rejected_alias_conflict": a_res.rejected_alias_conflict,
        "a_candidates_rejected_gain_non_positive": a_res.rejected_gain_non_positive,
        "a_candidates_rejected_context_rescore_negative": a_res.rejected_context_rescore_negative,
        "a_candidates_selected_final": a_res.selected,
        "stage3_ab_a_raw_token_len_summary_json": json.dumps(
            a_res.a_raw_token_len_summary, ensure_ascii=False
        ),
        "stage3_ab_a_alias_token_len_summary_json": json.dumps(
            a_res.a_alias_token_len_summary, ensure_ascii=False
        ),
        "stage3_ab_a_context_delta_summary_json": json.dumps(
            a_res.a_context_delta_summary, ensure_ascii=False
        ),
        "b_free_text_candidates_total": b_res.b_free_text_candidates_total,
        "b_free_text_candidates_visible_after_stage2": b_res.b_free_text_candidates_visible_after_stage2,
        "b_clusters_formed": b_res.b_clusters_formed,
        "b_clusters_rejected_too_small": b_res.b_clusters_rejected_too_small,
        "b_clusters_rejected_similarity_or_quality": b_res.b_clusters_rejected_similarity_or_quality,
        "b_clusters_rejected_intro_cost": b_res.b_clusters_rejected_intro_cost,
        "b_clusters_selected_final": b_res.b_clusters_selected_final,
        "stage3_ab_b_raw_total_summary_json": json.dumps(
            b_res.b_raw_total_summary, ensure_ascii=False
        ),
        "stage3_ab_b_code_total_summary_json": json.dumps(
            b_res.b_code_total_summary, ensure_ascii=False
        ),
        "stage3_ab_b_intro_cost_summary_json": json.dumps(
            b_res.b_intro_cost_summary, ensure_ascii=False
        ),
        "stage3_ab_b_candidates": b_res.candidates,
        "stage3_ab_b_cluster_count": b_res.cluster_count,
        "stage3_ab_b_used_clusters": b_res.used_clusters,
        "stage3_ab_b_intro_tokens": b_res.intro_tokens,
        "stage3_ab_b_sequence_saved": b_res.sequence_saved,
        "stage3_ab_b_effective_net_saving": b_res.effective_net_saving,
        "stage3_ab_b_fallback_count": b_res.fallback_count,
        "stage3_ab_b_avg_similarity": b_res.avg_similarity,
        "stage3_ab_b_risk_reject_count": b_res.risk_reject_count,
        "stage3_ab_b_intro_not_worth_count": b_res.intro_not_worth_count,
        "stage3_ab_b_reject_reason_counts": dict(b_res.reject_reason_counts),
        "stage3_ab_similarity_kind": b_res.similarity_kind,
        "stage3_ab_b_mode": b_res.mode,
        "stage3_ab_mode": conf.mode,
        "stage3_ab_vocab_entries": a_res.vocab_entries + b_res.vocab_entries,
        "stage3_ab_a_processing_mode": conf.a_processing_mode,
        "stage3_ab_a_cost_mode": conf.a_cost_mode,
        "stage3_ab_b_channel_priority": conf.b_channel_priority,
        **guard_telem,
    }
    return HybridABResult(
        encoded_text=final.encoded_text,
        a=a_res,
        b=b_res,
        fallback_count=b_res.fallback_count,
        meta=meta,
    )


def summary_dict(result: HybridABResult) -> dict[str, Any]:
    d = dict(result.meta)
    d["stage3_ab_a_entries_json"] = [asdict(e) for e in result.a.entries]
    d["stage3_ab_b_clusters_json"] = [asdict(c) for c in result.b.clusters]
    return d


class HybridABStage3Backend:
    name = "hybrid_ab"

    def encode(self, text: str, repo_config: Any, *, tokenizer: Any, tok_type: Optional[str]) -> Stage3EncodeResult:
        if tokenizer is None or not tok_type:
            return Stage3EncodeResult(encoded_text=text, vocab_entries=[], metrics={})
        cfg_raw = dict(getattr(repo_config, "stage3_ab_summary", {}) or {})
        if not cfg_raw:
            meta = {
                "stage3_ab_mode": "exact_only",
                "stage3_ab_similarity_kind": "lexical_bow_cosine",
                "stage3_ab_b_mode": "lexical_free_text_baseline",
                "stage3_ab_runtime_warning": "missing stage3_ab_summary; stage3 skipped",
                "stage3_ab_vocab_entries": [],
            }
            return Stage3EncodeResult(encoded_text=text, vocab_entries=[], metrics=meta)
        mode = str(cfg_raw.get("mode", "exact_only")).strip().lower()
        if mode not in {"exact_only", "hybrid"}:
            mode = "exact_only"

        def _truthy(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in {"1", "true", "yes", "on"}

        conf = HybridABConfig(
            mode=mode,
            free_text_min_chars=int(cfg_raw["free_text_min_chars"]),
            free_text_min_words=int(cfg_raw["free_text_min_words"]),
            key_like_patterns=tuple(cfg_raw.get("key_like_patterns", []) or ()),
            short_string_policy=str(cfg_raw.get("short_string_policy", "exact_candidate")),
            enable_mid_free_text=_truthy(cfg_raw.get("enable_mid_free_text", False)),
            free_text_mid_min_chars=int(cfg_raw.get("free_text_mid_min_chars", 14)),
            free_text_mid_min_words=int(cfg_raw.get("free_text_mid_min_words", 3)),
            allow_multiline_whitelist=_truthy(cfg_raw.get("allow_multiline_whitelist", False)),
            multiline_max_lines=int(cfg_raw.get("multiline_max_lines", 3)),
            multiline_max_chars=int(cfg_raw.get("multiline_max_chars", 220)),
            a_min_occ=int(cfg_raw["a_min_occ"]),
            a_min_net_gain=int(cfg_raw["a_min_net_gain"]),
            a_alias_style=str(cfg_raw["a_alias_style"]),
            a_alias_candidate_style=str(cfg_raw.get("a_alias_candidate_style", "token_cost_sorted")),
            b_similarity_threshold=float(cfg_raw["b_similarity_threshold"]),
            b_risk_threshold=float(cfg_raw["b_risk_threshold"]),
            b_min_cluster_size=int(cfg_raw["b_min_cluster_size"]),
            b_similarity_kind=str(cfg_raw.get("b_similarity_kind", "lexical_bow_cosine")),
            b_lexical_weight=float(cfg_raw.get("b_lexical_weight", 0.7)),
            b_char_weight=float(cfg_raw.get("b_char_weight", 0.3)),
            b_char_ngram_n=int(cfg_raw.get("b_char_ngram_n", 3)),
            a_processing_mode=str(cfg_raw.get("a_processing_mode", "full")).strip().lower(),
            a_cost_mode=str(cfg_raw.get("a_cost_mode", "local")).strip().lower(),
            enable_global_guardrail=_truthy(cfg_raw.get("enable_global_guardrail", False)),
            enable_incremental_rollback=_truthy(cfg_raw.get("enable_incremental_rollback", False)),
            min_raw_token_len=int(cfg_raw.get("min_raw_token_len", 1)),
            max_alias_token_len=int(cfg_raw.get("max_alias_token_len", 32)),
            context_window_chars=int(cfg_raw.get("context_window_chars", 80)),
            b_channel_priority=str(cfg_raw.get("b_channel_priority", "normal")).strip().lower(),
            a_alias_rank_pool_cap=int(cfg_raw.get("a_alias_rank_pool_cap", 32)),
            a_context_gain_margin=int(cfg_raw.get("a_context_gain_margin", 0)),
            a_enable_local_combo_greedy=_truthy(cfg_raw.get("a_enable_local_combo_greedy", False)),
            a_combo_max_additions=int(cfg_raw.get("a_combo_max_additions", 24)),
        )
        if mode != "hybrid" or not _truthy(cfg_raw.get("enable_b", False)):
            conf.mode = "exact_only"
        res = encode_stage3_hybrid_ab(text, tokenizer=tokenizer, tok_type=tok_type, cfg=conf)
        meta = summary_dict(res)
        vocab_entries = meta.get("stage3_ab_vocab_entries", []) or []
        return Stage3EncodeResult(encoded_text=res.encoded_text, vocab_entries=vocab_entries, metrics=meta)

    def compute_intro_cost(self, result: Stage3EncodeResult, *, tokenizer: Any, tok_type: Optional[str]) -> int:
        return compute_vocab_intro_cost(result.vocab_entries, mode=VOCAB_COST_MODE, tokenizer=tokenizer, tok_type=tok_type)
