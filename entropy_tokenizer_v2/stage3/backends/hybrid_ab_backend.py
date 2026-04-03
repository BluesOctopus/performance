from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from config import VOCAB_COST_MODE
from placeholder_accounting import compute_vocab_intro_cost
from stage3.backends.base import Stage3EncodeResult
from stage3.exact.alias_codec import ACodecResult, decode_exact_aliases, encode_exact_aliases
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
        "stage3_ab_a_protected_name_count": a_res.protected_name_count,
        "stage3_ab_a_min_occ_reject_count": a_res.min_occ_reject_count,
        "stage3_ab_a_net_gain_reject_count": a_res.net_gain_reject_count,
        "stage3_ab_a_reject_reason_counts": dict(a_res.reject_reason_counts),
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
    }
    return HybridABResult(
        encoded_text=b_res.encoded_text,
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
        )
        if mode != "hybrid" or not _truthy(cfg_raw.get("enable_b", False)):
            conf.mode = "exact_only"
        res = encode_stage3_hybrid_ab(text, tokenizer=tokenizer, tok_type=tok_type, cfg=conf)
        meta = summary_dict(res)
        vocab_entries = meta.get("stage3_ab_vocab_entries", []) or []
        return Stage3EncodeResult(encoded_text=res.encoded_text, vocab_entries=vocab_entries, metrics=meta)

    def compute_intro_cost(self, result: Stage3EncodeResult, *, tokenizer: Any, tok_type: Optional[str]) -> int:
        return compute_vocab_intro_cost(result.vocab_entries, mode=VOCAB_COST_MODE, tokenizer=tokenizer, tok_type=tok_type)

