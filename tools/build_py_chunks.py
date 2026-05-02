from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_chunks import build_chunks
from tokenizer_utils import resolve_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build syntax-aware Python chunks for offline diagnostics."
    )
    parser.add_argument("--input", required=True, help="Python file or directory to scan.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--tokenizer", default="gpt4", help="Tokenizer name: gpt4, cl100k_base, or Qwen/Qwen2.5-Coder-1.5B.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved = resolve_tokenizer(args.tokenizer)
    records = build_chunks(
        input_path,
        tokenizer_name=resolved.tokenizer_name,
        tok_type=resolved.tok_type,
        encoder=resolved.encoder,
    )
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
