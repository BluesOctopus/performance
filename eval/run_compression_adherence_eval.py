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
    total = len(rows)
    compressed_parse_ok = 0
    signature_ok = 0
    static_decode_ok = 0
    static_ast_equivalent = 0
    invalid_python = 0
    positive_gain = 0
    for row in rows:
        compressed_text = str(row.get("compressed_text", ""))
        try:
            ast.parse(compressed_text)
            compressed_parse_ok += 1
        except SyntaxError:
            invalid_python += 1
        alpha_metadata = dict(row.get("alpha_metadata", {}) or {})
        signature_ok += int(bool(alpha_metadata.get("alpha_public_signature_preserved", False)))
        static_metadata = dict(row.get("static_vocab_metadata", {}) or {})
        static_decode_ok += int(bool(static_metadata.get("decode_success", True)))
        static_ast_equivalent += int(bool(static_metadata.get("ast_equivalent", True)))
        positive_gain += int(float(row.get("effective_saved", 0) or 0) > 0)

    summary = {
        "compressed_parse_ok": compressed_parse_ok / total if total else 0.0,
        "alpha_public_signature_preserved": signature_ok / total if total else 0.0,
        "static_vocab_decode_success": static_decode_ok / total if total else 0.0,
        "static_vocab_ast_equivalent": static_ast_equivalent / total if total else 0.0,
        "invalid_python_rate": invalid_python / total if total else 0.0,
        "compression_gain_positive_rate": positive_gain / total if total else 0.0,
        "row_count": total,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
