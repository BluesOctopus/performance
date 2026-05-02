from __future__ import annotations

from typing import Any

from stage3.alpha_rename import AlphaPassResult, apply_alpha_rename_pass


def run_stage2_alpha_pass(
    stage2_text: str,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
) -> AlphaPassResult:
    return apply_alpha_rename_pass(
        stage2_text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
    )
