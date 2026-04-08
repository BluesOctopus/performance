"""Eval v2 on ``data/starcoder_1m_tokens.txt``; writes ``results/v2_compression_report.csv``."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import bootstrap_v2

bootstrap_v2.ensure()


def load_local_samples(sample_file: Path) -> list[str]:
    text = sample_file.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"<\|sample_\d+\|>\n", text)
    return [s for s in parts if s.strip()]


DEFAULT_TOKENIZERS = ("gpt4", "codegen", "santacoder")

CSV_ROW_ORDER = ("gpt4", "gpt2", "codegen", "santacoder")


def _merge_eval_outputs(results, configs, *, tokenizer_keys: list[str]) -> None:
    """Merge partial tokenizer runs into existing CSV/JSON."""
    import csv
    import json
    from dataclasses import asdict

    from config import RESULTS_DIR
    from v2_eval import EvalResult, save_results

    full = set(tokenizer_keys) == set(DEFAULT_TOKENIZERS) and len(tokenizer_keys) == len(
        DEFAULT_TOKENIZERS
    )
    if full:
        save_results(results, configs)
        return

    fields = [f.name for f in EvalResult.__dataclass_fields__.values()]
    float_keys = (
        "reduction_pct",
        "syntax_pct",
        "cleaning_pct",
        "replacement_pct",
        "baseline_bpb",
        "final_bpb",
        "baseline_entropy",
    )

    csv_path = RESULTS_DIR / "v2_compression_report.csv"
    by_key: dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_key[row["tokenizer_key"]] = row
    for r in results:
        row = asdict(r)
        for k in float_keys:
            row[k] = f"{row[k]:.6f}"
        by_key[r.tokenizer_key] = row
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in CSV_ROW_ORDER:
            if key in by_key:
                w.writerow(by_key[key])
                written.add(key)
        for key in sorted(by_key.keys() - written):
            w.writerow(by_key[key])
    print(f"\n[eval] CSV merged -> {csv_path}")

    detail_path = RESULTS_DIR / "v2_eval_detail.json"
    new_results = [asdict(r) for r in results]
    new_top = {k: cfg.scores_summary for k, cfg in configs.items()}
    new_skel = {k: cfg.selected_skeletons for k, cfg in configs.items()}
    if detail_path.exists():
        with open(detail_path, encoding="utf-8") as f:
            old = json.load(f)
        res_by = {d["tokenizer_key"]: d for d in old.get("results", [])}
        for d in new_results:
            res_by[d["tokenizer_key"]] = d
        order = [k for k in CSV_ROW_ORDER if k in res_by]
        order.extend(sorted(k for k in res_by if k not in order))
        old["results"] = [res_by[k] for k in order]
        ts = old.get("top_scores_by_tokenizer") or {}
        ts.update(new_top)
        old["top_scores_by_tokenizer"] = ts
        sk = old.get("selected_skeletons_by_tokenizer") or {}
        sk.update(new_skel)
        old["selected_skeletons_by_tokenizer"] = sk
        out = old
    else:
        out = {
            "results": new_results,
            "top_scores_by_tokenizer": new_top,
            "selected_skeletons_by_tokenizer": new_skel,
        }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[eval] Detail JSON merged -> {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="v2 全链路评估：starcoder 约 1M tokens 本地样本")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=list(DEFAULT_TOKENIZERS),
        metavar="KEY",
        help="config.EVAL_TOKENIZERS 中的键（默认：三个都跑）",
    )
    args = parser.parse_args()

    code_dir = bootstrap_v2.CODE_DIR

    from config import EVAL_TOKENIZERS
    from repo_miner import mine_from_sources
    from v2_eval import evaluate, print_report

    sample_file = code_dir / "data" / "starcoder_1m_tokens.txt"
    if not sample_file.exists():
        raise FileNotFoundError(f"缺少样本文件: {sample_file}")
    sources = load_local_samples(sample_file)
    print("Loaded samples:", len(sources))

    tokenizer_keys = args.tokenizers

    results = []
    configs = {}

    for key in tokenizer_keys:
        cfg = EVAL_TOKENIZERS.get(key)
        if cfg is None:
            print(f"[skip] 未知 tokenizer 键: {key!r}（见 config.EVAL_TOKENIZERS）")
            continue
        print("\n=== TOKENIZER:", key, "===")
        repo_cfg = mine_from_sources(
            sources=sources,
            tokenizer_key=key,
            tokenizer_cfg=cfg,
            cache_name="starcoderdata_1m_" + key,
            cache=True,
            verbose=True,
        )
        configs[key] = repo_cfg
        res = evaluate(sources, repo_cfg, key, cfg)
        results.append(res)

    print_report(results)
    _merge_eval_outputs(results, configs, tokenizer_keys=tokenizer_keys)


if __name__ == "__main__":
    main()
