from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from stage3.alpha_rename import AlphaPassResult, apply_alpha_rename_pass
from tokenizer_utils import count_tokens


@dataclass(frozen=True)
class FinalizedAlphaOutput:
    compressed_text: str
    alpha_metadata: dict[str, object]
    safety_checks: dict[str, bool]


def safe_parse_ok(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def safe_compile_ok(text: str) -> bool:
    try:
        compile(text, "<alpha_pipeline>", "exec")
    except SyntaxError:
        return False
    return True


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


def finalize_alpha_output(
    stage2_text: str,
    alpha_result: AlphaPassResult,
    *,
    encoder: Any,
    tok_type: str,
) -> FinalizedAlphaOutput:
    metadata = dict(alpha_result.metadata.to_dict())
    rollback_reason = str(metadata.get("alpha_rollback_reason", "") or metadata.get("alpha_skipped_reason", ""))
    if not rollback_reason:
        rollback_reason = "unknown_alpha_guardrail"

    delta_tokens = int(metadata.get("alpha_delta_tokens", 0) or 0)
    candidate_is_safe = (
        bool(metadata.get("alpha_ast_ok", False))
        and bool(metadata.get("alpha_compile_ok", False))
        and bool(metadata.get("alpha_public_signature_preserved", False))
        and delta_tokens > 0
        and bool(metadata.get("alpha_applied", False))
    )

    final_text = alpha_result.output_text if candidate_is_safe else stage2_text
    final_parse_ok = safe_parse_ok(final_text)
    final_compile_ok = safe_compile_ok(final_text)
    final_signature_ok = True if final_text == stage2_text else bool(metadata.get("alpha_public_signature_preserved", False))

    if candidate_is_safe:
        final_metadata = metadata
    else:
        final_tokens = count_tokens(stage2_text, encoder=encoder, tok_type=tok_type)
        final_metadata = {
            **metadata,
            "alpha_applied": False,
            "alpha_renamed_count": 0,
            "alpha_tokens": final_tokens,
            "alpha_delta_tokens": 0,
            "alpha_guardrail_triggered": True,
            "alpha_rollback_reason": rollback_reason,
            "alpha_skipped_reason": str(metadata.get("alpha_skipped_reason", "") or rollback_reason),
        }

    return FinalizedAlphaOutput(
        compressed_text=final_text,
        alpha_metadata=final_metadata,
        safety_checks={
            "compressed_parse_ok": final_parse_ok,
            "compressed_compile_ok": final_compile_ok,
            "alpha_public_signature_preserved": final_signature_ok,
        },
    )
