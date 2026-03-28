"""Paper-style tokenizer sweep on Starcoder 1M samples → CSV + ``docs/PAPER_TABLE_BENCHMARK.md``."""
from __future__ import annotations

import csv
import re
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import bootstrap_v2

bootstrap_v2.ensure()

import os

from config import HF_TOKEN, RESULTS_DIR

os.environ.setdefault("HF_TOKEN", HF_TOKEN)

from repo_miner import mine_from_sources
from v2_eval import EvalResult, evaluate

CODE_DIR = bootstrap_v2.CODE_DIR

PAPER_ROWS: list[tuple[str, str, dict]] = [
    ("codebert", "CodeBERT", {"type": "hf", "name": "microsoft/codebert-base"}),
    ("gpt2", "GPT-2", {"type": "hf", "name": "gpt2"}),
    ("codellama", "CodeLlama", {"type": "hf", "name": "codellama/CodeLlama-7b-hf"}),
    ("wizardcoder", "WizardCoder", {"type": "hf", "name": "WizardLM/WizardCoder-Python-7B-V1.0"}),
    ("deepseek-coder", "DeepSeek-Coder", {"type": "hf", "name": "deepseek-ai/deepseek-coder-1.3b-base"}),
    ("codegen", "CodeGen", {"type": "hf", "name": "Salesforce/codegen-350M-mono"}),
    ("codet5p", "CodeT5+", {"type": "hf", "name": "Salesforce/codet5p-220m-py"}),
    ("codex", "Codex", {"alias_of": "codegen"}),
    ("codet5", "CodeT5", {"type": "hf", "name": "Salesforce/codet5-base"}),
    ("starcoder", "StarCoder", {"type": "hf", "name": "bigcode/starcoder"}),
    ("santacoder", "SantaCoder", {"type": "hf", "name": "bigcode/santacoder"}),
    ("replit-code", "Replit-code", {"type": "hf", "name": "replit/replit-code-v1-3b"}),
    ("gpt-3.5", "GPT-3.5", {"type": "tiktoken", "tiktoken_model": "gpt-3.5-turbo"}),
    ("gpt4", "GPT-4", {"type": "tiktoken", "tiktoken_model": "gpt-4"}),
]

VOCAB_SOURCE: dict[str, str] = {
    "codebert": "Code",
    "gpt2": "Web",
    "codellama": "Web",
    "wizardcoder": "Web",
    "deepseek-coder": "Web",
    "codegen": "Web",
    "codet5p": "Web",
    "codex": "Web",
    "codet5": "Code",
    "starcoder": "Code",
    "santacoder": "Code",
    "replit-code": "Code",
    "gpt-3.5": "Web",
    "gpt4": "Web",
}


def load_samples() -> list[str]:
    p = CODE_DIR / "data" / "starcoder_1m_tokens.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"<\|sample_\d+\|>\n", text)
    return [s for s in parts if s.strip()]


def main() -> None:
    sources = load_samples()
    print(f"Loaded samples: {len(sources)}")

    results_ok: dict[str, EvalResult] = {}
    skipped: list[tuple[str, str]] = []

    for key, display, spec in PAPER_ROWS:
        if "alias_of" in spec:
            src = spec["alias_of"]
            if src not in results_ok:
                skipped.append((key, f"alias 目标 {src!r} 未成功"))
                print(f"[SKIP] {display}: alias {src!r} missing")
                continue
            results_ok[key] = replace(results_ok[src], tokenizer_key=key)
            print(f"[alias] {display} <- {src}")
            continue

        cfg = {k: v for k, v in spec.items()}
        try:
            print(f"\n{'='*60}\n  {display}  ({key})\n{'='*60}")
            repo = mine_from_sources(
                sources=sources,
                tokenizer_key=key,
                tokenizer_cfg=cfg,
                cache_name=f"paper_tbl_1m_{key}",
                cache=True,
                verbose=True,
            )
            res = evaluate(sources, repo, key, cfg)
            results_ok[key] = res
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            skipped.append((key, msg))
            print(f"[SKIP] {display}: {msg}")
            traceback.print_exc()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "paper_table_starcoder_1m.csv"
    base_fields = [f.name for f in EvalResult.__dataclass_fields__.values()]
    extra = ["paper_display_name", "vocab_source"]
    float_keys = (
        "reduction_pct",
        "syntax_pct",
        "cleaning_pct",
        "replacement_pct",
        "baseline_bpb",
        "final_bpb",
        "baseline_entropy",
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + extra)
        w.writeheader()
        for key, display, _spec in PAPER_ROWS:
            if key not in results_ok:
                continue
            r = results_ok[key]
            row = asdict(r)
            row["tokenizer_key"] = key
            for fk in float_keys:
                row[fk] = f"{row[fk]:.6f}"
            row["paper_display_name"] = display
            row["vocab_source"] = VOCAB_SOURCE.get(key, "")
            w.writerow(row)

    md_path = Path(__file__).resolve().parent.parent / "docs" / "PAPER_TABLE_BENCHMARK.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Starcoder ~1M samples × paper-style tokenizers (v2 three-stage)",
        "",
        "- Samples: `performance/code/data/starcoder_1m_tokens.txt` (301 chunks).",
        "- Mining: `lossless_clean` keeps `#` comments and docstrings.",
        "- Reduction % = (baseline − final) / baseline for this pipeline.",
        "",
        "| Tokenizer | Vocab 来源 | V₀ | Baseline tokens | Final tokens | 降幅 |",
        "|-----------|------------|----|-----------------|--------------|------|",
    ]
    for key, display, _spec in PAPER_ROWS:
        if key not in results_ok:
            continue
        r = results_ok[key]
        lines.append(
            f"| {display} | {VOCAB_SOURCE.get(key, '')} | {r.V0:,} | "
            f"{r.baseline_tokens:,} | {r.final_tokens:,} | {r.reduction_pct:.2f}% |"
        )
    lines.extend(["", "## 跳过或失败", ""])
    if skipped:
        for k, msg in skipped:
            lines.append(f"- `{k}`: {msg}")
    else:
        lines.append("- （无）")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[done] CSV -> {csv_path}")
    print(f"[done] MD  -> {md_path}")


if __name__ == "__main__":
    main()
