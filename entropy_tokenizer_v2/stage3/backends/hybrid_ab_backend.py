from __future__ import annotations

from typing import Any, Optional

from config import VOCAB_COST_MODE
from placeholder_accounting import compute_vocab_intro_cost
from stage3.backends.base import Stage3EncodeResult
from stage3.hybrid_ab import HybridABConfig, encode_stage3_hybrid_ab, summary_dict


class HybridABStage3Backend:
    name = "hybrid_ab"

    def encode(
        self,
        text: str,
        repo_config: Any,
        *,
        tokenizer: Any,
        tok_type: Optional[str],
    ) -> Stage3EncodeResult:
        if tokenizer is None or not tok_type:
            return Stage3EncodeResult(encoded_text=text, vocab_entries=[], meta={})

        cfg_raw = dict(getattr(repo_config, "stage3_ab_summary", {}) or {})
        if not cfg_raw:
            # Preserve previous behavior: avoid hidden defaults, and write a warning for eval.
            meta = {
                "stage3_ab_mode": "exact_only",
                "stage3_ab_similarity_kind": "lexical_bow_cosine",
                "stage3_ab_b_mode": "disabled",
                "stage3_ab_runtime_warning": "missing stage3_ab_summary; stage3 skipped",
                "stage3_ab_vocab_entries": [],
            }
            setattr(repo_config, "_stage3_hybrid_last_meta", meta)
            return Stage3EncodeResult(encoded_text=text, vocab_entries=[], meta=meta)

        mode = str(cfg_raw.get("mode", "exact_only")).strip().lower()
        if mode not in {"exact_only", "hybrid"}:
            mode = "exact_only"

        conf = HybridABConfig(
            mode=mode,
            free_text_min_chars=int(cfg_raw["free_text_min_chars"]),
            free_text_min_words=int(cfg_raw["free_text_min_words"]),
            key_like_patterns=tuple(cfg_raw.get("key_like_patterns", []) or ()),
            a_min_occ=int(cfg_raw["a_min_occ"]),
            a_min_net_gain=int(cfg_raw["a_min_net_gain"]),
            a_alias_style=str(cfg_raw["a_alias_style"]),
            b_similarity_threshold=float(cfg_raw["b_similarity_threshold"]),
            b_risk_threshold=float(cfg_raw["b_risk_threshold"]),
            b_min_cluster_size=int(cfg_raw["b_min_cluster_size"]),
        )
        if mode != "hybrid" or not bool(cfg_raw.get("enable_b", False)):
            conf.mode = "exact_only"

        res = encode_stage3_hybrid_ab(text, tokenizer=tokenizer, tok_type=tok_type, cfg=conf)
        meta = summary_dict(res)
        setattr(repo_config, "_stage3_hybrid_last_meta", meta)

        vocab_entries = meta.get("stage3_ab_vocab_entries", []) or []
        return Stage3EncodeResult(encoded_text=res.encoded_text, vocab_entries=vocab_entries, meta=meta)

    def compute_intro_cost(
        self,
        result: Stage3EncodeResult,
        *,
        tokenizer: Any,
        tok_type: Optional[str],
    ) -> int:
        return compute_vocab_intro_cost(
            result.vocab_entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )

