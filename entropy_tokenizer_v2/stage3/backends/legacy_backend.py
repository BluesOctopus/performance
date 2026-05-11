from __future__ import annotations

from typing import Any, Optional

from markers import get_syn_line_spans
from config import VOCAB_COST_MODE
from placeholder_accounting import compute_vocab_intro_cost
from stage3.backends.base import Stage3Backend, Stage3EncodeResult
from token_scorer import (
    apply_token_replacement_with_protected_spans,
    build_stage3_vocab_entries_from_used_placeholders,
    collect_used_stage3_placeholders,
)


class LegacyStage3Backend:
    name = "legacy"

    def encode(
        self,
        text: str,
        repo_config: Any,
        *,
        tokenizer: Any,
        tok_type: Optional[str],
    ) -> Stage3EncodeResult:
        rmap = getattr(repo_config, "replacement_map", None) or {}
        if not rmap:
            return Stage3EncodeResult(encoded_text=text, vocab_entries=[], metrics={})

        protected_spans = get_syn_line_spans(text)
        encoded = apply_token_replacement_with_protected_spans(text, rmap, protected_spans)
        used = collect_used_stage3_placeholders(encoded, rmap)
        entries = build_stage3_vocab_entries_from_used_placeholders(used)
        return Stage3EncodeResult(encoded_text=encoded, vocab_entries=entries, metrics={})

    def compute_intro_cost(
        self,
        result: Stage3EncodeResult,
        *,
        tokenizer: Any,
        tok_type: Optional[str],
    ) -> int:
        # Pipeline already passes tokenizer/tok_type; just reuse the canonical helper.
        return compute_vocab_intro_cost(
            result.vocab_entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )

