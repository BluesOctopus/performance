from __future__ import annotations

from typing import Any, Optional

from config import VOCAB_COST_MODE
from placeholder_accounting import (
    build_used_plan_a_vocab_entries,
    compute_vocab_intro_cost,
)
from repo_miner import load_plan_a_codebooks
from stage3.backends.base import Stage3EncodeResult
from stage3.literal_codec.pipeline.source_codec import (
    encode_python_source_plan_a,
    extract_used_plan_a_entries,
)


class PlanAStage3Backend:
    name = "plan_a"

    def encode(
        self,
        text: str,
        repo_config: Any,
        *,
        tokenizer: Any,
        tok_type: Optional[str],
    ) -> Stage3EncodeResult:
        books = load_plan_a_codebooks(repo_config)
        if not books:
            return Stage3EncodeResult(encoded_text=text, vocab_entries=[], meta={})

        esc = getattr(repo_config, "stage3_escape_prefix", "__L__")
        encoded = encode_python_source_plan_a(text, books, escape_prefix=esc)

        used = extract_used_plan_a_entries(encoded, books, esc)
        entries = build_used_plan_a_vocab_entries(books, used, escape_prefix=esc)
        meta = {
            "stage3_plan_a_used_count": len(used),
        }
        return Stage3EncodeResult(encoded_text=encoded, vocab_entries=entries, meta=meta)

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

