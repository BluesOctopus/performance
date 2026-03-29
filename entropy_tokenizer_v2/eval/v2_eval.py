"""Three-stage compress + aggregate token metrics (per file and corpus)."""

import csv
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

import bootstrap_v2

bootstrap_v2.ensure()


def _ensure_stage3_pkg() -> None:
    root = Path(__file__).resolve().parents[1] / "stage3"
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)

from config import (
    CACHE_DIR,
    EVAL_DATASET,
    EVAL_NUM_SAMPLES,
    EVAL_TOKENIZERS,
    HF_DISK_DATASET_FALLBACK,
    HF_TOKEN,
    RESULTS_DIR,
    STAGE2_DEFAULT_MODE,
    STAGE2_DEFAULT_PROFILE,
    VOCAB_COST_MODE,
)
from repo_miner import RepoConfig, _encode, _load_tokenizer, _vocab_size
from pipeline import CompressionBreakdown, apply_pipeline, _stage1_vocab_intro
from placeholder_accounting import compute_vocab_intro_cost
from token_scorer import (
    build_stage3_vocab_entries_from_used_placeholders,
    collect_used_stage3_placeholders,
)


def apply_v2_compression(
    source: str,
    repo_config: RepoConfig,
    tokenizer,
    tok_type: str,
    count_fn=None,
    stage2_profile: str = STAGE2_DEFAULT_PROFILE,
    stage2_mode: str = STAGE2_DEFAULT_MODE,
) -> tuple[str, CompressionBreakdown]:
    return apply_pipeline(
        source,
        repo_config,
        tokenizer,
        tok_type,
        count_fn=count_fn,
        stage2_profile=stage2_profile,
        stage2_mode=stage2_mode,
    )


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
    ds_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", EVAL_DATASET)
    cache_path = CACHE_DIR / f"eval_samples_{ds_tag}_{num_samples}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return samples[:num_samples]

    if HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    try:
        from datasets import load_dataset
        ds = load_dataset(EVAL_DATASET, split="train",
                          streaming=True, token=HF_TOKEN or None)
        samples = []
        for ex in ds:
            samples.append(ex["content"])
            if len(samples) >= num_samples:
                break
    except Exception:
        from datasets import load_from_disk
        ds = load_from_disk(str(HF_DISK_DATASET_FALLBACK))
        samples = ds["content"][:num_samples]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    return samples


@dataclass
class EvalResult:
    tokenizer_key:       str
    stage2_profile:      str
    stage2_mode:         str
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
    sequence_final_tokens: int
    sequence_reduction_pct: float
    stage1_vocab_intro_tokens: int
    stage2_vocab_intro_tokens: int
    stage3_vocab_intro_tokens: int
    final_vocab_intro_tokens: int
    effective_total_tokens: int
    effective_total_saved: int
    effective_total_reduction_pct: float
    syntax_pct:          float
    cleaning_pct:        float
    replacement_pct:     float
    baseline_bpb:        float
    final_bpb:           float
    baseline_entropy:    float
    V0:                  int
    k_star_syntax:       int
    n_replacement_words: int
    stage3_backend: str = "legacy"
    stage3_assignments: int = 0
    stage3_fields: str = ""
    stage3_dictionary_coverage: float = 0.0
    stage3_expected_gain: float = 0.0
    stage3_component_saved: int = 0
    stage3_used_entries: int = 0
    stage3_used_entries_variable: int = 0
    stage3_used_entries_attribute: int = 0
    stage3_used_entries_string: int = 0
    stage3_cost_model: str = ""
    stage3_plan_a_profile: str = ""
    stage3_assignment_by_field_json: str = ""


def evaluate(
    sources: list[str],
    repo_config: RepoConfig,
    tokenizer_key: str,
    tokenizer_cfg: dict,
    stage2_profile: str = STAGE2_DEFAULT_PROFILE,
    stage2_mode: str = STAGE2_DEFAULT_MODE,
) -> EvalResult:
    tokenizer, tok_type = _load_tokenizer(tokenizer_key, tokenizer_cfg)
    V0 = _vocab_size(tokenizer, tok_type)
    count_fn = None

    total_bytes = sum(len(s.encode("utf-8")) for s in sources)

    backend = getattr(repo_config, "stage3_backend", "legacy")
    plan_books = None
    plan_esc = getattr(repo_config, "stage3_escape_prefix", "__L__")
    if backend == "plan_a":
        from repo_miner import load_plan_a_codebooks

        plan_books = load_plan_a_codebooks(repo_config)

    agg = CompressionBreakdown(
        baseline_tokens=0, after_syntax=0,
        after_cleaning=0, after_replacement=0,
    )
    baseline_token_counts: Counter = Counter()
    legacy_stage3_used: list[str] = []
    legacy_seen: set[str] = set()
    plan_a_used_union: set[tuple[str, str]] = set()

    for src in tqdm(sources, desc=f"  [{tokenizer_key}] compressing", leave=False):
        compressed, fr = apply_v2_compression(
            src,
            repo_config,
            tokenizer,
            tok_type,
            count_fn,
            stage2_profile=stage2_profile,
            stage2_mode=stage2_mode,
        )

        for tok_id in _encode(tokenizer, tok_type, src):
            baseline_token_counts[tok_id] += 1

        agg.baseline_tokens   += fr.baseline_tokens
        agg.after_syntax      += fr.after_syntax
        agg.after_cleaning    += fr.after_cleaning
        agg.after_replacement += fr.after_replacement

        if backend == "legacy" and repo_config.replacement_map:
            for ph in collect_used_stage3_placeholders(compressed, repo_config.replacement_map):
                if ph not in legacy_seen:
                    legacy_seen.add(ph)
                    legacy_stage3_used.append(ph)
        elif backend == "plan_a" and plan_books:
            _ensure_stage3_pkg()
            from literal_codec.pipeline.source_codec import extract_used_plan_a_entries

            plan_a_used_union |= extract_used_plan_a_entries(compressed, plan_books, plan_esc)

    B = agg.baseline_tokens
    F_seq = agg.after_replacement
    F = max(1, F_seq)

    sm = getattr(repo_config, "stage3_plan_a_summary", None) or {}
    stage3_assignments = int(sm.get("stage3_plan_a_assignments_count", 0))
    stage3_fields = ",".join(sm.get("stage3_plan_a_fields", []) or [])
    stage3_dictionary_coverage = float(sm.get("stage3_plan_a_dictionary_coverage_mean", 0.0))
    stage3_expected_gain = float(sm.get("stage3_plan_a_total_expected_gain_sum", 0.0))
    stage3_cost_model = str(sm.get("stage3_plan_a_cost_model", "") or "")
    stage3_plan_a_profile = str(sm.get("stage3_plan_a_profile", "") or "")
    stage3_assignment_by_field_json = json.dumps(
        sm.get("stage3_plan_a_assignment_by_field", {}), ensure_ascii=False
    )

    if backend == "plan_a":
        from placeholder_accounting import build_used_plan_a_vocab_entries
        from repo_miner import load_plan_a_codebooks

        books = plan_books if plan_books else load_plan_a_codebooks(repo_config)
        entries = build_used_plan_a_vocab_entries(
            books,
            plan_a_used_union,
            escape_prefix=plan_esc,
        )
        stage3_vocab_intro_tokens = compute_vocab_intro_cost(
            entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
    else:
        entries = build_stage3_vocab_entries_from_used_placeholders(legacy_stage3_used)
        stage3_vocab_intro_tokens = compute_vocab_intro_cost(
            entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )

    used_v = used_a = used_s = 0
    if backend == "plan_a":
        for fld, _code in plan_a_used_union:
            if fld == "variable":
                used_v += 1
            elif fld == "attribute":
                used_a += 1
            elif fld == "string":
                used_s += 1

    s1_intro = _stage1_vocab_intro(repo_config, tokenizer, tok_type)
    s2_intro = 0
    final_vocab_intro = s1_intro + s2_intro + stage3_vocab_intro_tokens
    effective_total = F_seq + final_vocab_intro
    effective_saved = B - effective_total
    seq_red = (1.0 - F_seq / B) * 100.0 if B else 0.0
    eff_red = (1.0 - effective_total / B) * 100.0 if B else 0.0

    return EvalResult(
        tokenizer_key       = tokenizer_key,
        stage2_profile      = stage2_profile,
        stage2_mode         = stage2_mode,
        n_files             = len(sources),
        baseline_tokens     = B,
        syntax_tokens       = agg.after_syntax,
        cleaning_tokens     = agg.after_cleaning,
        final_tokens        = F_seq,
        syntax_saved        = agg.syntax_saved,
        cleaning_saved      = agg.cleaning_saved,
        replacement_saved   = agg.replacement_saved,
        total_saved         = agg.total_saved,
        reduction_pct       = seq_red,
        sequence_final_tokens=F_seq,
        sequence_reduction_pct=seq_red,
        stage1_vocab_intro_tokens=s1_intro,
        stage2_vocab_intro_tokens=s2_intro,
        stage3_vocab_intro_tokens=stage3_vocab_intro_tokens,
        final_vocab_intro_tokens=final_vocab_intro,
        effective_total_tokens=effective_total,
        effective_total_saved=effective_saved,
        effective_total_reduction_pct=eff_red,
        syntax_pct          = (agg.syntax_saved   / B) * 100.0 if B else 0.0,
        cleaning_pct        = (agg.cleaning_saved / B) * 100.0 if B else 0.0,
        replacement_pct     = (agg.replacement_saved / B) * 100.0 if B else 0.0,
        baseline_bpb        = _bpb(B, V0, total_bytes),
        final_bpb           = _bpb(F, V0, total_bytes),
        baseline_entropy    = _entropy(baseline_token_counts),
        V0                  = V0,
        k_star_syntax       = len(repo_config.selected_skeletons),
        n_replacement_words = len(repo_config.replacement_map),
        stage3_backend      = backend,
        stage3_assignments  = stage3_assignments,
        stage3_fields       = stage3_fields,
        stage3_dictionary_coverage = stage3_dictionary_coverage,
        stage3_expected_gain = stage3_expected_gain,
        stage3_component_saved = agg.replacement_saved,
        stage3_used_entries=len(plan_a_used_union) if backend == "plan_a" else 0,
        stage3_used_entries_variable=used_v,
        stage3_used_entries_attribute=used_a,
        stage3_used_entries_string=used_s,
        stage3_cost_model=stage3_cost_model,
        stage3_plan_a_profile=stage3_plan_a_profile,
        stage3_assignment_by_field_json=stage3_assignment_by_field_json,
    )


def print_report(results: list[EvalResult]):
    w = 168
    print("\n" + "=" * w)
    print("  v2 compression — evaluation report")
    print("=" * w)

    hdr = (
        f"  {'Tokenizer':<20} {'Profile':<18} {'Mode':<9} {'Baseline':>10} "
        f"{'SeqFinal':>10} {'Seq%':>7} {'EffTotal':>10} {'Eff%':>7} "
        f"{'Vocab':>8} {'S1voc':>6} {'S3voc':>6} "
        f"{'Syn%':>6} {'Cln%':>6} {'Tok%':>6} "
        f"{'BPB_b':>8} {'BPB_f':>8} {'K*':>4} {'Nrp':>5} {'S3be':>8}"
    )
    print(hdr)
    print("-" * w)

    for r in results:
        print(
            f"  {r.tokenizer_key:<20} {r.stage2_profile:<18} {r.stage2_mode:<9} "
            f"{r.baseline_tokens:>10,} {r.sequence_final_tokens:>10,} "
            f"{r.sequence_reduction_pct:>6.1f}% {r.effective_total_tokens:>10,} "
            f"{r.effective_total_reduction_pct:>6.1f}% "
            f"{r.final_vocab_intro_tokens:>8,} {r.stage1_vocab_intro_tokens:>6,} "
            f"{r.stage3_vocab_intro_tokens:>6,} "
            f"{r.syntax_pct:>5.1f}% {r.cleaning_pct:>5.1f}% {r.replacement_pct:>5.1f}% "
            f"{r.baseline_bpb:>8.4f} {r.final_bpb:>8.4f} "
            f"{r.k_star_syntax:>4} {r.n_replacement_words:>5} "
            f"{r.stage3_backend:>8}"
        )

    print("=" * w)
    print(
        "  SeqFinal / Seq% = corpus sequence tokens only (placeholders as 1); "
        "EffTotal / Eff% = SeqFinal + corpus-once vocab intro (S1+S2+S3); "
        "Vocab = S1voc+S2voc+S3voc (S2voc=0 here); S3voc = Stage3 intro only; "
        "Syn%/Cln%/Tok% = stage1/2/3 sequence savings vs baseline; "
        "Nrp = legacy replacement_map size (0 for plan_a)."
    )


def save_results(
    results: list[EvalResult],
    repo_config_by_tok: dict,
    *,
    csv_name: str = "v2_compression_report.csv",
    detail_name: str = "v2_eval_detail.json",
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / csv_name
    fields = [f.name for f in EvalResult.__dataclass_fields__.values()]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            for k in (
                "reduction_pct",
                "sequence_reduction_pct",
                "effective_total_reduction_pct",
                "syntax_pct",
                "cleaning_pct",
                "replacement_pct",
                "baseline_bpb",
                "final_bpb",
                "baseline_entropy",
                "stage3_dictionary_coverage",
                "stage3_expected_gain",
            ):
                row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"\n[eval] CSV saved → {csv_path}")

    detail_path = RESULTS_DIR / detail_name
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
        "stage3_plan_a_summary_by_tokenizer": {
            tok_key: getattr(cfg, "stage3_plan_a_summary", {})
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
    stage2_profile: str = STAGE2_DEFAULT_PROFILE,
    stage2_mode: str = STAGE2_DEFAULT_MODE,
    *,
    stage3_backend: str | None = None,
    output_csv: str = "v2_compression_report.csv",
    output_detail: str = "v2_eval_detail.json",
) -> list[EvalResult]:
    from config import STAGE3_BACKEND
    from repo_miner import mine_from_sources

    backend = (stage3_backend or STAGE3_BACKEND).strip().lower()
    if backend not in {"legacy", "plan_a"}:
        backend = "legacy"

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
            cache_name=f"eval_{num_samples}_{stage2_profile}_{stage2_mode}",
            cache=True,
            verbose=verbose,
            stage3_backend=backend,
        )
        repo_config_by_tok[tok_key] = repo_config

        result = evaluate(
            samples,
            repo_config,
            tok_key,
            cfg,
            stage2_profile=stage2_profile,
            stage2_mode=stage2_mode,
        )
        all_results.append(result)

        if verbose:
            print(
                f"  Sequence reduction: {result.sequence_reduction_pct:.1f}%  "
                f"(Syntax {result.syntax_pct:.1f}% + "
                f"Clean {result.cleaning_pct:.1f}% + "
                f"Token {result.replacement_pct:.1f}%)  |  "
                f"Effective-total reduction: {result.effective_total_reduction_pct:.1f}%  "
                f"(vocab intro {result.final_vocab_intro_tokens:,} tok)"
            )

    print_report(all_results)
    save_results(all_results, repo_config_by_tok, csv_name=output_csv, detail_name=output_detail)
    return all_results
