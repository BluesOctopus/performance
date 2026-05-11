"""CLI: ``eval`` (HF samples or ``--repo``), ``demo`` (single file / toy). From repo: ``python .../eval/run_v2.py eval``."""

import argparse
import os
import sys
from pathlib import Path

import bootstrap_v2

bootstrap_v2.ensure()

from config import (
    EVAL_NUM_SAMPLES,
    EVAL_TOKENIZERS,
    STAGE3_BACKEND,
)


def cmd_eval(args):
    from v2_eval import run_evaluation

    if getattr(args, "stage2_hybrid_ab_profile", None):
        os.environ["ET_STAGE2_HYBRID_AB_PROFILE"] = str(args.stage2_hybrid_ab_profile)
    if getattr(args, "stage2_hybrid_ab_mode", None):
        os.environ["ET_STAGE2_HYBRID_AB_MODE"] = str(args.stage2_hybrid_ab_mode)

    tok_keys = args.tokenizers if args.tokenizers else None

    if args.repo:
        # Evaluate using a local repo as both mining corpus AND eval corpus
        from repo_miner import collect_py_sources, mine_from_repo_path
        from v2_eval import evaluate, print_report, save_results

        print(f"[run_v2] Evaluating local repo: {args.repo}")
        sources = collect_py_sources(args.repo)
        if not sources:
            print("ERROR: no .py files found.", file=sys.stderr)
            sys.exit(1)
        if args.samples:
            sources = sources[: args.samples]

        results = []
        configs = {}
        backend = args.stage3_backend if args.stage3_backend is not None else STAGE3_BACKEND
        tag = getattr(args, "stage3_output_tag", "") or ""
        csv_name = f"v2_compression_report{tag}.csv" if tag else "v2_compression_report.csv"
        detail_name = f"v2_eval_detail{tag}.json" if tag else "v2_eval_detail.json"
        for tok_key in (tok_keys or list(EVAL_TOKENIZERS.keys())):
            cfg = EVAL_TOKENIZERS.get(tok_key)
            if cfg is None:
                print(f"Unknown tokenizer '{tok_key}', skipping")
                continue
            repo_config = mine_from_repo_path(
                args.repo, tok_key, cfg, cache=True, stage3_backend=backend
            )
            configs[tok_key] = repo_config
            r = evaluate(
                sources,
                repo_config,
                tok_key,
                cfg,
                stage2_profile=args.stage2_profile,
                stage2_mode=args.stage2_mode,
            )
            results.append(r)

        print_report(results)
        save_results(results, configs, csv_name=csv_name, detail_name=detail_name)

    else:
        # Use the HF eval dataset
        n = args.samples or EVAL_NUM_SAMPLES
        backend = args.stage3_backend if args.stage3_backend is not None else STAGE3_BACKEND
        tag = getattr(args, "stage3_output_tag", "") or ""
        csv_name = f"v2_compression_report{tag}.csv" if tag else "v2_compression_report.csv"
        detail_name = f"v2_eval_detail{tag}.json" if tag else "v2_eval_detail.json"
        run_evaluation(
            tokenizer_keys=tok_keys,
            num_samples=n,
            stage2_profile=args.stage2_profile,
            stage2_mode=args.stage2_mode,
            stage3_backend=backend,
            output_csv=csv_name,
            output_detail=detail_name,
        )


def cmd_demo(args):
    """
    Show the compression effect on a single Python file (or built-in toy code).
    Runs with the first available tokenizer unless --tokenizer is specified.
    """
    tok_key = args.tokenizer or list(EVAL_TOKENIZERS.keys())[0]
    cfg = EVAL_TOKENIZERS.get(tok_key)
    if cfg is None:
        print(f"ERROR: unknown tokenizer '{tok_key}'", file=sys.stderr)
        sys.exit(1)

    if args.file:
        source = Path(args.file).read_text(encoding="utf-8", errors="replace")
    else:
        source = _TOY_CODE

    print(f"\n[demo] Tokenizer: {tok_key}")
    print(f"[demo] Source ({len(source)} chars):\n{'─'*60}")
    print(source[:2000] + ("..." if len(source) > 2000 else ""))
    print("─" * 60)

    from repo_miner import mine_from_sources, _load_tokenizer
    from v2_eval import apply_v2_compression

    tokenizer, tok_type = _load_tokenizer(tok_key, cfg)

    repo_config = mine_from_sources(
        sources=[source],
        tokenizer_key=tok_key,
        tokenizer_cfg=cfg,
        cache_name=f"demo_{hash(source) & 0xFFFF:04x}",
        cache=True,
        verbose=True,
        min_freq=1,
        stage3_backend=args.stage3_backend if args.stage3_backend is not None else STAGE3_BACKEND,
    )

    compressed, fr = apply_v2_compression(source, repo_config, tokenizer, tok_type)

    print(f"\n[demo] Compressed output:\n{'─'*60}")
    print(compressed[:2000] + ("..." if len(compressed) > 2000 else ""))
    print("─" * 60)

    B = fr.baseline_tokens
    print(f"\n{'─'*60}")
    print(f"  Baseline tokens      : {B:,}")
    print(f"  After Stage-1 syntax : {fr.after_syntax:,}  "
          f"(-{fr.syntax_saved:,}, {fr.syntax_saved/B*100:.1f}%)")
    print(f"  After Stage-2 clean  : {fr.after_cleaning:,}  "
          f"(-{fr.cleaning_saved:,}, {fr.cleaning_saved/B*100:.1f}%)")
    print(f"  After Stage-3 tokens : {fr.after_replacement:,}  "
          f"(-{fr.replacement_saved:,}, {fr.replacement_saved/B*100:.1f}%)")
    print("-" * 51)
    print(f"  Total reduction      : -{fr.total_saved:,} tokens  "
          f"({fr.total_saved/B*100:.1f}%)")
    print("-" * 60 + "\n")


_TOY_CODE = '''\
"""Utility functions for processing data files."""

import os
import json
from typing import Optional, List

# Default output directory
DEFAULT_OUTPUT_DIR = "/tmp/output"


def load_json_file(filepath: str) -> Optional[dict]:
    """Load a JSON file and return its contents as a dictionary."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    with open(filepath, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data


def process_records(records: List[dict], output_dir: str = DEFAULT_OUTPUT_DIR) -> int:
    """
    Process a list of records and write results to output_dir.
    Returns the number of records successfully processed.
    """
    success_count = 0
    for record in records:
        record_id = record.get("id", "unknown")
        record_name = record.get("name", "")
        if not record_name:
            continue
        output_path = os.path.join(output_dir, f"{record_id}.json")
        try:
            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(record, out_file, indent=2)
            success_count += 1
        except OSError as error_obj:
            print(f"Failed to write {output_path}: {error_obj}")
    return success_count


class DataProcessor:
    """Handles batch processing of data records."""

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processed = 0
        self.errors = 0

    def run(self) -> bool:
        """Run the processing pipeline."""
        os.makedirs(self.output_dir, exist_ok=True)
        for filename in os.listdir(self.input_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.input_dir, filename)
            data = load_json_file(filepath)
            if data is None:
                self.errors += 1
                continue
            records = data.get("records", [])
            count = process_records(records, self.output_dir)
            self.processed += count
        return self.errors == 0
'''


def main():
    parser = argparse.ArgumentParser(
        prog="run_v2",
        description="entropy_tokenizer_v2 eval / demo",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run full evaluation")
    p_eval.add_argument("--repo", type=str, default=None,
                        help="Path to local Python repo (default: use HF dataset)")
    p_eval.add_argument("--samples", type=int, default=None,
                        help=f"Number of samples (default: {EVAL_NUM_SAMPLES})")
    p_eval.add_argument("--tokenizers", nargs="+", default=None,
                        help="Tokenizer keys to evaluate (default: all)")
    p_eval.add_argument(
        "--stage2-profile",
        type=str,
        default=None,
        help=(
            "Stage2 profile (omit for backend defaults: hybrid_ab uses ET_STAGE2_HYBRID_AB_PROFILE; "
            "others use ET_STAGE2_DEFAULT_PROFILE)"
        ),
    )
    p_eval.add_argument(
        "--stage2-mode",
        type=str,
        default=None,
        choices=["linewise", "blockwise"],
        help=(
            "Stage2 mode (omit for backend defaults: hybrid_ab uses ET_STAGE2_HYBRID_AB_MODE; "
            "others use ET_STAGE2_DEFAULT_MODE)"
        ),
    )
    p_eval.add_argument(
        "--stage2-hybrid-ab-profile",
        type=str,
        default=None,
        help="Set ET_STAGE2_HYBRID_AB_PROFILE for this run (hybrid_ab implicit Stage2 only).",
    )
    p_eval.add_argument(
        "--stage2-hybrid-ab-mode",
        type=str,
        default=None,
        choices=["linewise", "blockwise"],
        help="Set ET_STAGE2_HYBRID_AB_MODE for this run (hybrid_ab implicit Stage2 only).",
    )
    p_eval.add_argument(
        "--stage3-backend",
        type=str,
        default=None,
        choices=["legacy", "plan_a", "hybrid_ab"],
        help="Stage3 backend (default: ET_STAGE3_BACKEND / config)",
    )
    p_eval.add_argument(
        "--stage3-output-tag",
        type=str,
        default="",
        help="Optional suffix for v2_compression_report*.csv and v2_eval_detail*.json",
    )
    p_eval.add_argument(
        "--stage3-use-tiktoken",
        action="store_true",
        help="Reserved; Plan A uses the same tokenizer as eval for consistent costs",
    )

    p_demo = sub.add_parser("demo", help="Show compression on a single file")
    p_demo.add_argument("--file", type=str, default=None,
                        help="Python source file to compress (default: built-in toy code)")
    p_demo.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer key (default: first in config)")
    p_demo.add_argument(
        "--stage3-backend",
        type=str,
        default=None,
        choices=["legacy", "plan_a", "hybrid_ab"],
        help="Stage3 backend (default: ET_STAGE3_BACKEND / config)",
    )

    args = parser.parse_args()

    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
