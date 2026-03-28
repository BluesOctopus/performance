"""
Main pipeline: data → mine → eval.

Usage:
    python run_pipeline.py --all              # Full pipeline
    python run_pipeline.py --all --quick      # Quick test (~10 MB data)
    python run_pipeline.py --step data        # Only download/cache data
    python run_pipeline.py --step mine        # Only run operator mining
    python run_pipeline.py --step eval        # Only run compression eval
    python run_pipeline.py --step eval --tokenizers gpt4 santacoder
    python run_pipeline.py --step eval --budgets 20 50 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def step_data(args):
    from data_loader import stream_and_save
    stream_and_save(force=args.force, quick=args.quick)


def step_mine(args):
    from frequency_miner import run_mining
    max_files = 500 if args.quick else None
    run_mining(max_files=max_files)


def step_eval(args):
    from compress_eval import run_evaluation
    run_evaluation(
        budgets=args.budgets,
        tokenizer_keys=args.tokenizers,
    )


STEPS = {"data": step_data, "mine": step_mine, "eval": step_eval}


def main():
    parser = argparse.ArgumentParser(
        description="Operator-Based Hierarchical Tokenizer Pipeline"
    )
    parser.add_argument(
        "--step", type=str, default=None,
        choices=list(STEPS.keys()),
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tokenizers", nargs="+", type=str, default=None)
    parser.add_argument("--budgets", nargs="+", type=int, default=None)

    args = parser.parse_args()

    if not args.step and not args.all:
        parser.print_help()
        return

    steps_to_run = [args.step] if args.step else list(STEPS.keys())

    t_total = time.time()
    for name in steps_to_run:
        print(f"\n{'='*60}")
        print(f"  STEP: {name.upper()}")
        print(f"{'='*60}\n")
        t0 = time.time()
        try:
            STEPS[name](args)
        except Exception as e:
            print(f"\n[ERROR] Step '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
            if args.step:
                sys.exit(1)
            continue
        print(f"\n[pipeline] '{name}' done in {time.time()-t0:.1f}s")

    print(f"\n[pipeline] Total: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
