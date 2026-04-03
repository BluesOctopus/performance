"""Stage3 Hybrid AB codec orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .aliasing.alias_codec import ACodecResult, decode_exact_aliases, encode_exact_aliases
from .router import ABRoutingConfig
from .semantic_cluster.semantic_codec import (
    BCodecResult,
    SemanticClassifierConfig,
    encode_semantic_strings,
)


@dataclass(slots=True)
class HybridABConfig:
    mode: str = "exact_only"
    # Router
    free_text_min_chars: int = 24
    free_text_min_words: int = 4
    key_like_patterns: tuple[str, ...] = ()
    # Exact channel
    a_min_occ: int = 2
    a_min_net_gain: int = 1
    a_alias_style: str = "short"
    # Semantic channel
    b_similarity_threshold: float = 0.82
    b_risk_threshold: float = 0.72
    b_min_cluster_size: int = 2


@dataclass(slots=True)
class HybridABResult:
    encoded_text: str
    a: ACodecResult
    b: BCodecResult
    fallback_count: int
    meta: dict[str, Any] = field(default_factory=dict)


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
        key_like_patterns=conf.key_like_patterns or ABRoutingConfig().key_like_patterns,
    )
    a_res = encode_exact_aliases(
        text,
        tokenizer=tokenizer,
        tok_type=tok_type,
        route_cfg=route_cfg,
        min_occ=conf.a_min_occ,
        min_net_gain=conf.a_min_net_gain,
        alias_style=conf.a_alias_style,
    )
    if conf.mode == "hybrid":
        b_res = encode_semantic_strings(
            a_res.encoded_text,
            tokenizer=tokenizer,
            tok_type=tok_type,
            similarity_threshold=conf.b_similarity_threshold,
            risk_threshold=conf.b_risk_threshold,
            min_cluster_size=conf.b_min_cluster_size,
            classifier_cfg=SemanticClassifierConfig(
                free_text_min_chars=conf.free_text_min_chars,
                free_text_min_words=conf.free_text_min_words,
            ),
        )
    else:
        b_res = BCodecResult(encoded_text=a_res.encoded_text)
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
        "stage3_ab_b_candidates": b_res.candidates,
        "stage3_ab_b_cluster_count": b_res.cluster_count,
        "stage3_ab_b_used_clusters": b_res.used_clusters,
        "stage3_ab_b_intro_tokens": b_res.intro_tokens,
        "stage3_ab_b_sequence_saved": b_res.sequence_saved,
        "stage3_ab_b_effective_net_saving": b_res.effective_net_saving,
        "stage3_ab_b_fallback_count": b_res.fallback_count,
        "stage3_ab_b_avg_similarity": b_res.avg_similarity,
        "stage3_ab_b_risk_reject_count": b_res.risk_reject_count,
        "stage3_ab_similarity_kind": "lexical_bow_cosine",
        "stage3_ab_b_mode": "lexical_free_text_baseline" if conf.mode == "hybrid" else "disabled",
        "stage3_ab_mode": conf.mode,
        "stage3_ab_vocab_entries": a_res.vocab_entries + b_res.vocab_entries,
    }
    fallback = b_res.fallback_count
    return HybridABResult(
        encoded_text=b_res.encoded_text,
        a=a_res,
        b=b_res,
        fallback_count=fallback,
        meta=meta,
    )


def decode_stage3_hybrid_ab_a_only(text: str, result: HybridABResult) -> str:
    """Decode exact A aliases only (B is intentionally lossy)."""
    return decode_exact_aliases(text, result.a.entries)


def summary_dict(result: HybridABResult) -> dict[str, Any]:
    d = dict(result.meta)
    d["stage3_ab_a_entries_json"] = [asdict(e) for e in result.a.entries]
    d["stage3_ab_b_clusters_json"] = [asdict(c) for c in result.b.clusters]
    return d
