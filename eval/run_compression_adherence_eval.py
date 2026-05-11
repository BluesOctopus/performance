from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data-level compression adherence checks.")
    parser.add_argument("--manifest", required=True, help="Input manifest jsonl")
    parser.add_argument("--out", required=True, help="Output json summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = build_summary(rows)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    if total == 0:
        return {
            "row_count": 0,
            "variant_distribution": {},
            "invalid_python_rate": 0.0,
            "decoded_invalid_python_rate": 0.0,
            "alpha_public_signature_preserved": 0.0,
            "compression_gain_positive_rate": 0.0,
            "encoded_parse_ok": 0.0,
            "static_vocab_decode_success": 0.0,
            "decoded_parse_ok": 0.0,
            "decoded_compile_ok": 0.0,
            "decoded_ast_equivalent": 0.0,
            "invalid_python_count": 0,
            "signature_failure_count": 0,
            "alpha_guardrail_rollback_count": 0,
            "alpha_no_gain_count": 0,
            "alpha_parse_failure_count": 0,
            "alpha_compile_failure_count": 0,
            "static_decode_failure_count": 0,
            "decoded_parse_failure_count": 0,
            "avg_raw_tokens": 0.0,
            "avg_compressed_tokens": 0.0,
            "avg_effective_saved": 0.0,
        }

    variant_distribution: dict[str, int] = {}
    compressed_parse_ok = 0
    signature_ok = 0
    static_decode_ok = 0
    encoded_parse_ok = 0
    decoded_parse_ok = 0
    decoded_compile_ok = 0
    decoded_ast_equivalent = 0
    invalid_python_count = 0
    decoded_invalid_python_count = 0
    signature_failure_count = 0
    alpha_guardrail_rollback_count = 0
    alpha_no_gain_count = 0
    alpha_parse_failure_count = 0
    alpha_compile_failure_count = 0
    static_decode_failure_count = 0
    decoded_parse_failure_count = 0
    positive_gain = 0

    for row in rows:
        variant = str(row.get("variant", ""))
        variant_distribution[variant] = variant_distribution.get(variant, 0) + 1

        compressed_text = str(row.get("compressed_text", ""))
        compressed_ok = parse_ok(compressed_text)
        compressed_parse_ok += int(compressed_ok)

        alpha_metadata = dict(row.get("alpha_metadata", {}) or {})
        signature_preserved = bool(alpha_metadata.get("alpha_public_signature_preserved", False))
        signature_ok += int(signature_preserved)
        signature_failure_count += int(not signature_preserved)
        alpha_guardrail_rollback_count += int(bool(alpha_metadata.get("alpha_guardrail_triggered", False)))
        rollback_reason = str(alpha_metadata.get("alpha_rollback_reason", "") or alpha_metadata.get("alpha_skipped_reason", ""))
        alpha_no_gain_count += int(rollback_reason == "no_token_gain")
        alpha_parse_failure_count += int(
            rollback_reason in {"parse_failed", "ast_parse_failed_after_rename", "rename_broke_parseability"}
            or not bool(alpha_metadata.get("alpha_ast_ok", False))
        )
        alpha_compile_failure_count += int(
            rollback_reason == "compile_failed_after_rename"
            or not bool(alpha_metadata.get("alpha_compile_ok", False))
        )

        static_metadata = dict(row.get("static_vocab_metadata", {}) or {})
        is_static_variant = variant == "stage2_alpha_stage1_static"
        static_decode_success = bool(static_metadata.get("decode_success", not is_static_variant))
        static_decode_ok += int(static_decode_success)
        static_decode_failure_count += int(is_static_variant and not static_decode_success)

        encoded_ok = bool(static_metadata.get("encoded_parse_ok", compressed_ok))
        encoded_parse_ok += int(encoded_ok)
        decoded_ok = bool(static_metadata.get("decoded_parse_ok", compressed_ok if not is_static_variant else False))
        decoded_parse_ok += int(decoded_ok)
        decoded_compile = bool(static_metadata.get("decoded_compile_ok", compressed_ok if not is_static_variant else False))
        decoded_compile_ok += int(decoded_compile)
        decoded_ast_ok = bool(static_metadata.get("decoded_ast_equivalent", True if not is_static_variant else False))
        decoded_ast_equivalent += int(decoded_ast_ok)

        invalid_python_count += int(not compressed_ok)
        decoded_invalid = is_static_variant and (not decoded_ok or not decoded_compile)
        decoded_invalid_python_count += int(decoded_invalid)
        decoded_parse_failure_count += int(is_static_variant and not decoded_ok)
        positive_gain += int(float(row.get("effective_saved", 0) or 0) > 0)

    uses_decoded_gate = any(str(row.get("variant", "")) == "stage2_alpha_stage1_static" for row in rows)
    invalid_rate = decoded_invalid_python_count / total if uses_decoded_gate else invalid_python_count / total
    summary = {
        "row_count": total,
        "variant_distribution": variant_distribution,
        "compressed_parse_ok": compressed_parse_ok / total,
        "alpha_public_signature_preserved": signature_ok / total,
        "compression_gain_positive_rate": positive_gain / total,
        "encoded_parse_ok": encoded_parse_ok / total,
        "static_vocab_decode_success": static_decode_ok / total,
        "decoded_parse_ok": decoded_parse_ok / total,
        "decoded_compile_ok": decoded_compile_ok / total,
        "decoded_ast_equivalent": decoded_ast_equivalent / total,
        "invalid_python_rate": invalid_rate,
        "decoded_invalid_python_rate": decoded_invalid_python_count / total,
        "invalid_python_count": decoded_invalid_python_count if uses_decoded_gate else invalid_python_count,
        "signature_failure_count": signature_failure_count,
        "alpha_guardrail_rollback_count": alpha_guardrail_rollback_count,
        "alpha_no_gain_count": alpha_no_gain_count,
        "alpha_parse_failure_count": alpha_parse_failure_count,
        "alpha_compile_failure_count": alpha_compile_failure_count,
        "static_decode_failure_count": static_decode_failure_count,
        "decoded_parse_failure_count": decoded_parse_failure_count,
        "avg_raw_tokens": sum(float(row.get("raw_tokens", 0) or 0) for row in rows) / total,
        "avg_compressed_tokens": sum(float(row.get("compressed_tokens", 0) or 0) for row in rows) / total,
        "avg_effective_saved": sum(float(row.get("effective_saved", 0) or 0) for row in rows) / total,
    }
    return summary


def parse_ok(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
