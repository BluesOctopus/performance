"""Three-stage compress + aggregate token metrics (per file and corpus)."""

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

from tqdm.auto import tqdm

import bootstrap_v2

bootstrap_v2.ensure()

from config import (
    CACHE_DIR, EVAL_DATASET, EVAL_NUM_SAMPLES,
    EVAL_TOKENIZERS, HF_TOKEN, RESULTS_DIR,
)
from lossy_cleaner import CleaningConfig, clean_code
from repo_miner import RepoConfig, _encode, _load_tokenizer, _vocab_size
from syntax_compressor import compress_source_syntax
from token_scorer import apply_token_replacement
from marker_count import RE_ALL_MARKERS, count_augmented

_SYN_LINE_RE = re.compile(r'^\s*<SYN_\d+>\b')


def _is_syn_line(line: str) -> bool:
    return bool(_SYN_LINE_RE.match(line))


def _clean_stage2_skip_syn(text: str) -> str:
    """Stage 2 on non-SYN lines only; SYN lines get ``rstrip`` only."""
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=True,
        remove_trailing_whitespace=True,
        remove_docstrings=False,
        remove_indentation=True,
    )

    out_lines: list[str] = []
    for line in text.splitlines():
        if _is_syn_line(line):
            out_lines.append(line.rstrip())
            continue

        cleaned_line, _ = clean_code(line, cfg)
        if cleaned_line.strip():
            out_lines.append(cleaned_line)

    return "\n".join(out_lines)


def _replace_stage3_skip_syn(text: str, rmap: dict[str, str]) -> str:
    """Stage 3 only on lines that are not ``<SYN_n>`` headers."""
    if not rmap:
        return text

    out_lines: list[str] = []
    for line in text.splitlines():
        if _is_syn_line(line):
            out_lines.append(line)
        else:
            out_lines.append(apply_token_replacement(line, rmap))
    return "\n".join(out_lines)


def _count_with_ops(text: str, tokenizer, tok_type: str) -> int:
    return count_augmented(text, tokenizer, tok_type, pattern=RE_ALL_MARKERS)


@dataclass
class FileResult:
    baseline_tokens:   int
    after_syntax:      int
    after_cleaning:    int
    after_replacement: int

    @property
    def syntax_saved(self) -> int:
        return self.baseline_tokens - self.after_syntax

    @property
    def cleaning_saved(self) -> int:
        return self.after_syntax - self.after_cleaning

    @property
    def replacement_saved(self) -> int:
        return self.after_cleaning - self.after_replacement

    @property
    def total_saved(self) -> int:
        return self.baseline_tokens - self.after_replacement


def apply_v2_compression(
    source: str,
    repo_config: RepoConfig,
    tokenizer,
    tok_type: str,
    count_fn=None,
) -> tuple[str, FileResult]:
    if count_fn is None:
        def count_fn_local(text: str) -> int:
            return _count_with_ops(text, tokenizer, tok_type)
        count_fn = count_fn_local

    baseline_tokens = count_fn(source)

    skeletons = repo_config.skeleton_candidates()
    after_s1 = compress_source_syntax(source, skeletons)
    after_s1_tokens = count_fn(after_s1)

    after_s2 = _clean_stage2_skip_syn(after_s1)
    after_s2_tokens = count_fn(after_s2)

    rmap = repo_config.replacement_map
    after_s3 = _replace_stage3_skip_syn(after_s2, rmap)
    after_s3_tokens = count_fn(after_s3)

    result = FileResult(
        baseline_tokens=baseline_tokens,
        after_syntax=after_s1_tokens,
        after_cleaning=after_s2_tokens,
        after_replacement=after_s3_tokens,
    )
    return after_s3, result


def _entropy(token_counts: Counter) -> float:
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for cnt in token_counts.values():
        if cnt > 0:
            p = cnt / total
            h -= p * math.log2(p)
    return h


def _bpb(total_tokens: int, vocab_size: int, total_bytes: int) -> float:
    if total_bytes == 0:
        return 0.0
    return total_tokens * math.log2(vocab_size) / total_bytes


def load_eval_samples(num_samples: int = EVAL_NUM_SAMPLES) -> list[str]:
    cache_path = CACHE_DIR / "eval_100star_samples.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return samples[:num_samples]

    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        from datasets import load_dataset
        ds = load_dataset(EVAL_DATASET, split="train",
                          streaming=True, token=HF_TOKEN)
        samples = []
        for ex in ds:
            samples.append(ex["content"])
            if len(samples) >= num_samples:
                break
    except Exception:
        from datasets import load_from_disk
        ds = load_from_disk(str(CACHE_DIR.parent / "data" / "test"))
        samples = ds["content"][:num_samples]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    return samples


@dataclass
class EvalResult:
    tokenizer_key:       str
    n_files:             int
    baseline_tokens:     int
    syntax_tokens:       int
    cleaning_tokens:     int
    final_tokens:        int
    syntax_saved:        int
    cleaning_saved:      int
    replacement_saved:   int
    total_saved:         int
    reduction_pct:       float
    syntax_pct:          float
    cleaning_pct:        float
    replacement_pct:     float
    baseline_bpb:        float
    final_bpb:           float
    baseline_entropy:    float
    V0:                  int
    k_star_syntax:       int
    n_replacement_words: int


def evaluate(
    sources: list[str],
    repo_config: RepoConfig,
    tokenizer_key: str,
    tokenizer_cfg: dict,
) -> EvalResult:
    tokenizer, tok_type = _load_tokenizer(tokenizer_key, tokenizer_cfg)
    V0 = _vocab_size(tokenizer, tok_type)
    count_fn = None

    total_bytes = sum(len(s.encode("utf-8")) for s in sources)

    agg = FileResult(
        baseline_tokens=0, after_syntax=0,
        after_cleaning=0, after_replacement=0,
    )
    baseline_token_counts: Counter = Counter()

    for src in tqdm(sources, desc=f"  [{tokenizer_key}] compressing", leave=False):
        _, fr = apply_v2_compression(src, repo_config, tokenizer, tok_type, count_fn)

        for tok_id in _encode(tokenizer, tok_type, src):
            baseline_token_counts[tok_id] += 1

        agg.baseline_tokens   += fr.baseline_tokens
        agg.after_syntax      += fr.after_syntax
        agg.after_cleaning    += fr.after_cleaning
        agg.after_replacement += fr.after_replacement

    B = agg.baseline_tokens
    F = max(1, agg.after_replacement)

    return EvalResult(
        tokenizer_key       = tokenizer_key,
        n_files             = len(sources),
        baseline_tokens     = B,
        syntax_tokens       = agg.after_syntax,
        cleaning_tokens     = agg.after_cleaning,
        final_tokens        = F,
        syntax_saved        = agg.syntax_saved,
        cleaning_saved      = agg.cleaning_saved,
        replacement_saved   = agg.replacement_saved,
        total_saved         = agg.total_saved,
        reduction_pct       = (1.0 - F / B) * 100.0 if B else 0.0,
        syntax_pct          = (agg.syntax_saved   / B) * 100.0 if B else 0.0,
        cleaning_pct        = (agg.cleaning_saved / B) * 100.0 if B else 0.0,
        replacement_pct     = (agg.replacement_saved / B) * 100.0 if B else 0.0,
        baseline_bpb        = _bpb(B, V0, total_bytes),
        final_bpb           = _bpb(F, V0, total_bytes),
        baseline_entropy    = _entropy(baseline_token_counts),
        V0                  = V0,
        k_star_syntax       = len(repo_config.selected_skeletons),
        n_replacement_words = len(repo_config.replacement_map),
    )


def print_report(results: list[EvalResult]):
    w = 120
    print("\n" + "=" * w)
    print("  v2 compression — evaluation report")
    print("=" * w)

    hdr = (f"  {'Tokenizer':<20} {'Baseline':>10} {'Final':>10} "
           f"{'Total%':>7} {'Syntax%':>7} {'Clean%':>7} {'Token%':>7} "
           f"{'BPB_base':>9} {'BPB_final':>9} "
           f"{'K*_syn':>6} {'N_repl':>7}")
    print(hdr)
    print("-" * w)

    for r in results:
        print(
            f"  {r.tokenizer_key:<20} {r.baseline_tokens:>10,}   {r.final_tokens:>10,} "
            f"{r.reduction_pct:>6.1f}% {r.syntax_pct:>6.1f}% "
            f"{r.cleaning_pct:>6.1f}% {r.replacement_pct:>6.1f}% "
            f"{r.baseline_bpb:>9.4f} {r.final_bpb:>9.4f} "
            f"{r.k_star_syntax:>6} {r.n_replacement_words:>7}"
        )

    print("=" * w)
    print("  Syntax%/Clean%/Token% = stage1/2/3 savings vs baseline; K*_syn, N_repl = skeletons, replacement_map size")


def save_results(results: list[EvalResult], repo_config_by_tok: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "v2_compression_report.csv"
    fields = [f.name for f in EvalResult.__dataclass_fields__.values()]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            for k in ("reduction_pct", "syntax_pct", "cleaning_pct",
                      "replacement_pct", "baseline_bpb", "final_bpb",
                      "baseline_entropy"):
                row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"\n[eval] CSV saved → {csv_path}")

    detail_path = RESULTS_DIR / "v2_eval_detail.json"
    detail = {
        "results": [asdict(r) for r in results],
        "top_scores_by_tokenizer": {
            tok_key: cfg.scores_summary
            for tok_key, cfg in repo_config_by_tok.items()
        },
        "selected_skeletons_by_tokenizer": {
            tok_key: cfg.selected_skeletons
            for tok_key, cfg in repo_config_by_tok.items()
        },
    }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"[eval] Detail JSON saved → {detail_path}")


def run_evaluation(
    tokenizer_keys: Optional[list[str]] = None,
    num_samples: int = EVAL_NUM_SAMPLES,
    verbose: bool = True,
) -> list[EvalResult]:
    from repo_miner import mine_from_sources

    if tokenizer_keys is None:
        tokenizer_keys = list(EVAL_TOKENIZERS.keys())

    if verbose:
        print(f"[eval] Loading {num_samples} evaluation samples ...")
    samples = load_eval_samples(num_samples)
    if verbose:
        print(f"[eval] Loaded {len(samples)} samples  "
              f"({sum(len(s.encode()) for s in samples):,} bytes total)")

    all_results: list[EvalResult] = []
    repo_config_by_tok: dict[str, RepoConfig] = {}

    for tok_key in tokenizer_keys:
        cfg = EVAL_TOKENIZERS.get(tok_key)
        if cfg is None:
            print(f"[eval] Unknown tokenizer '{tok_key}', skipping")
            continue

        print(f"\n{'='*60}\n  TOKENIZER: {tok_key}\n{'='*60}")

        repo_config = mine_from_sources(
            sources=samples,
            tokenizer_key=tok_key,
            tokenizer_cfg=cfg,
            cache_name=f"eval_{num_samples}",
            cache=True,
            verbose=verbose,
        )
        repo_config_by_tok[tok_key] = repo_config

        result = evaluate(samples, repo_config, tok_key, cfg)
        all_results.append(result)

        if verbose:
            print(f"  Total reduction: {result.reduction_pct:.1f}%  "
                  f"(Syntax {result.syntax_pct:.1f}% + "
                  f"Clean {result.cleaning_pct:.1f}% + "
                  f"Token {result.replacement_pct:.1f}%)")

    print_report(all_results)
    save_results(all_results, repo_config_by_tok)
    return all_results
