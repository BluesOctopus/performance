from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_chunks import build_chunks
from tokenizer_utils import resolve_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample deduplicated real function chunks for offline diagnostics.")
    parser.add_argument("--input", required=True, help="Repository or corpus directory containing Python files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-functions", type=int, default=1000, help="Maximum number of functions")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B", help="Tokenizer name")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = resolve_tokenizer(args.tokenizer)
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(
        input_path,
        tokenizer_name=resolved.tokenizer_name,
        tok_type=resolved.tok_type,
        encoder=resolved.encoder,
    )
    filtered = []
    seen_keys: set[str] = set()
    for chunk in chunks:
        if chunk["symbol_type"] not in {"function", "async_function"}:
            continue
        if not dedent_parse_ok(str(chunk["chunk_text"])):
            continue
        raw_tokens = int(chunk["raw_tokens"])
        if raw_tokens < 20 or raw_tokens > 800:
            continue
        dedented = textwrap.dedent(str(chunk["chunk_text"])).strip()
        key = "\n".join(line.rstrip() for line in dedented.splitlines())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        row = dict(chunk)
        row["chunk_ast_parse_ok_after_dedent"] = True
        filtered.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(filtered)
    sampled = sorted(filtered[: args.max_functions], key=lambda row: (str(row["source_id"]), int(row["start_line"])))

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "tokenizer_name": resolved.tokenizer_name,
        "seed": args.seed,
        "max_functions": args.max_functions,
        "candidate_count": len(filtered),
        "sampled_count": len(sampled),
        "dedupe_key": "dedented_trimmed_source",
        "raw_token_range": [20, 800],
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def dedent_parse_ok(text: str) -> bool:
    try:
        ast.parse(textwrap.dedent(text))
    except SyntaxError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
