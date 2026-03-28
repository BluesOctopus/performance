"""
Operator-Based Compression Rate Evaluation with MDL Framework.

The operator selection problem is formalised as Minimum Description Length
(MDL) optimisation:

    L_total(D, S) = L_data(D | S) + L_codebook(S)
                  = N_compressed * log₂(V₀ + |S|) + Σ_k cb_k * log₂(V₀)

Greedy selection adds operators one at a time.  The exact marginal change
in total description length when adding operator k to set S of size |S| is:

    ΔL_k = (N − spi_k·count_k) · log₂(V₀+|S|+1)     ← data after savings
         − N · log₂(V₀+|S|)                           ← data before
         + cb_k · log₂(V₀)                            ← codebook entry

This captures three competing effects:
  • Savings: fewer tokens (spi_k · count_k removed)
  • Vocab expansion penalty: every surviving token costs more bits
    because the vocabulary grew from V₀+|S| to V₀+|S|+1
  • Codebook cost: describing the operator template in the base vocab

K* is the point where ΔL first becomes ≥ 0.

Pipeline:
  1. Load mined AST + lexical patterns from frequency_miner output.
  2. For each target tokenizer:
     a. Compute per-candidate savings, codebook cost, and token-level
        net benefit for initial ranking.
     b. Pre-compute baseline tokens and AST skeletons on eval data.
     c. Run greedy MDL selection with exact ΔL to find K*.
  3. Evaluate at fixed budgets + K*:
     a. Report L_total, token reduction, bits-per-byte, entropy.
  4. Compare with SimPy.
"""

import ast
import csv
import json
import math
import os
import re
from collections import Counter
from typing import Optional

import tiktoken
import transformers
from tqdm.auto import tqdm

from config import (
    EVAL_TOKENIZERS, EVAL_100STAR_NUM_SAMPLES, EVAL_100STAR_DATASET,
    CACHE_DIR, RESULTS_DIR, HF_TOKEN, DATA_DIR,
    OPERATOR_BUDGETS, SIMPY_REPORTED, CODEBOOK_OVERHEAD_PER_OP,
)
from frequency_miner import _anonymize_ast_node


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(key: str, cfg: dict):
    """Return (tokenizer_object, type_str)."""
    if cfg["type"] == "tiktoken":
        return tiktoken.encoding_for_model(cfg["tiktoken_model"]), "tiktoken"
    tok = transformers.AutoTokenizer.from_pretrained(
        cfg["name"], trust_remote_code=True, token=HF_TOKEN,
    )
    return tok, "hf"


def encode(tokenizer, tok_type: str, text: str) -> list[int]:
    if tok_type == "tiktoken":
        return tokenizer.encode(text, allowed_special="all")
    return tokenizer.encode(text, add_special_tokens=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Eval sample loading (reuses cached 100star samples)
# ═══════════════════════════════════════════════════════════════════════════════

def load_eval_samples(num_samples: int = EVAL_100STAR_NUM_SAMPLES) -> list[str]:
    cache_path = CACHE_DIR / "eval_100star_samples.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return samples[:num_samples]

    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        from datasets import load_dataset
        ds = load_dataset(
            EVAL_100STAR_DATASET, split="train",
            streaming=True, token=HF_TOKEN,
        )
        samples = []
        for ex in ds:
            samples.append(ex["content"])
            if len(samples) >= num_samples:
                break
    except Exception:
        from datasets import load_from_disk
        ds = load_from_disk(str(DATA_DIR / "test"))
        samples = ds["content"][:num_samples]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer-aware savings computation
# ═══════════════════════════════════════════════════════════════════════════════

def savings_for_ast_skeleton(skeleton: str, tokenizer, tok_type: str) -> int:
    """
    How many tokens are saved per instance if this AST skeleton becomes an operator.

    The 'fixed part' of the skeleton (everything except placeholder content)
    is currently encoded as multiple tokens; an operator replaces them all with 1.
    """
    fixed_text = re.sub(r'\{[0-9]+\}', '', skeleton)
    if not fixed_text.strip():
        return 0
    n_tokens = len(encode(tokenizer, tok_type, fixed_text))
    return max(0, n_tokens - 1)


def savings_for_lexical_pattern(pattern: str, tokenizer, tok_type: str) -> int:
    """
    How many tokens are saved per instance if this prefix/suffix becomes an operator.
    e.g. 'self.' encoded as 2 tokens → savings = 2 - 1 = 1
    """
    if not pattern.strip():
        return 0
    n_tokens = len(encode(tokenizer, tok_type, pattern))
    return max(0, n_tokens - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# MDL: codebook cost & optimal-K
# ═══════════════════════════════════════════════════════════════════════════════

def compute_codebook_cost(pattern: str, tokenizer, tok_type: str) -> int:
    """
    Number of tokens required to *describe* this operator in the codebook.

    Cost = tokens_to_encode(template) + CODEBOOK_OVERHEAD_PER_OP.
    The template is the pattern string itself (skeleton or prefix/suffix).
    """
    tpl_tokens = len(encode(tokenizer, tok_type, pattern))
    return tpl_tokens + CODEBOOK_OVERHEAD_PER_OP


def compute_eval_frequencies(
    candidates: list[dict],
    samples: list[str],
    ast_skeletons_list: list[Counter],
) -> dict[str, int]:
    """
    Count actual occurrences of every candidate pattern on the eval corpus.
    This gives consistent frequencies for MDL selection on the same data
    that will be used for evaluation.
    """
    eval_freq: dict[str, int] = {}

    ast_patterns = {c["pattern"] for c in candidates if c["type"] == "syntax"}
    for skel_counts in ast_skeletons_list:
        for skel, cnt in skel_counts.items():
            if skel in ast_patterns:
                eval_freq[skel] = eval_freq.get(skel, 0) + cnt

    lex_candidates = [
        (c["pattern"], c["type"])
        for c in candidates if c["type"] in ("prefix", "suffix")
    ]
    if lex_candidates:
        for code in tqdm(samples, desc="    eval-freq lexical", leave=False):
            for pattern, pat_type in lex_candidates:
                regex = _get_lexical_regex(pattern, pat_type)
                n = len(regex.findall(code))
                if n:
                    eval_freq[pattern] = eval_freq.get(pattern, 0) + n

    return eval_freq


def greedy_mdl_select(candidates: list[dict], N_baseline: int,
                      V0: int,
                      eval_freq: Optional[dict[str, int]] = None) -> int:
    """
    Greedy forward selection minimising MDL total description length.

    At each step k, compute the *exact* marginal change in bits:

        DeltaL_k = N_new * log2(V0+k) - N_current * log2(V0+k-1)
                   + cb_k * log2(V0)

    This captures three competing effects:
      - Savings: fewer tokens (spi_k * count_k removed from data)
      - Vocab expansion penalty: all surviving tokens need more bits
      - Codebook cost: describing the operator template in base vocab

    If eval_freq is provided, uses eval-corpus frequencies (consistent
    with the evaluation data).  Otherwise falls back to mining frequencies.

    Candidates must be pre-sorted by mdl_net_benefit descending.
    Returns K* -- the index where greedy stops (prefix of accepted ops).
    Each candidate is annotated with 'marginal_delta_bits'.
    """
    log2_V0 = math.log2(V0)
    N_current = N_baseline
    S_size = 0
    k_star = 0

    for c in candidates:
        spi = c["savings_per_instance"]
        count = eval_freq.get(c["pattern"], 0) if eval_freq else c["frequency"]
        cb = c["codebook_cost"]

        if count == 0:
            c["marginal_delta_bits"] = cb * log2_V0
            break

        log2_V_current = math.log2(V0 + S_size)
        log2_V_new = math.log2(V0 + S_size + 1)

        N_new = N_current - spi * count
        delta = N_new * log2_V_new - N_current * log2_V_current + cb * log2_V0

        c["marginal_delta_bits"] = delta

        if delta >= 0:
            break

        k_star = S_size + 1
        N_current = N_new
        S_size += 1

    return k_star


def get_vocab_size(tokenizer, tok_type: str) -> int:
    """Return the base vocabulary size V_0."""
    if tok_type == "tiktoken":
        return tokenizer.n_vocab
    return tokenizer.vocab_size


# ═══════════════════════════════════════════════════════════════════════════════
# Information-theoretic metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_token_entropy(token_counts: Counter) -> float:
    """
    Empirical Shannon entropy of a token frequency distribution (bits).

    H = -sum_t p(t) * log2(p(t))
    """
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in token_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def compute_bits_per_byte(total_tokens: int, vocab_size: int,
                          total_original_bytes: int) -> float:
    """
    Bits-per-byte: a normalised, tokenizer-agnostic compression metric.

    bpb = total_tokens * log2(vocab_size) / total_original_bytes
    """
    if total_original_bytes == 0:
        return 0.0
    return total_tokens * math.log2(vocab_size) / total_original_bytes


def compute_mdl_score(compressed_tokens: int, vocab_size_aug: int,
                      codebook_cost_tokens: int, vocab_size_base: int) -> float:
    """
    Total MDL description length (bits).

    L_total = L_data + L_codebook
            = compressed_tokens * log2(V_0 + |S|)
              + codebook_cost_tokens * log2(V_0)
    """
    l_data = compressed_tokens * math.log2(vocab_size_aug)
    l_codebook = codebook_cost_tokens * math.log2(vocab_size_base)
    return l_data + l_codebook


# ═══════════════════════════════════════════════════════════════════════════════
# Match counting on evaluation data
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_ast_skeletons(code: str) -> Counter:
    """
    Parse the code's AST once, generate skeleton for every statement node.
    Returns {skeleton: count} for ALL patterns (not filtered by selection).
    This allows caching — call once per file, then filter by budget later.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()

    counts: Counter = Counter()
    for node in ast.walk(tree):
        skeleton = _anonymize_ast_node(node)
        if skeleton:
            counts[skeleton] += 1
    return counts


_LEX_CACHE: dict[str, re.Pattern] = {}

def _get_lexical_regex(pattern: str, pat_type: str) -> re.Pattern:
    key = (pattern, pat_type)
    if key not in _LEX_CACHE:
        if pat_type == "prefix":
            _LEX_CACHE[key] = re.compile(
                rf'(?<![a-zA-Z0-9_]){re.escape(pattern)}[a-zA-Z0-9_]+'
            )
        else:
            _LEX_CACHE[key] = re.compile(
                rf'[a-zA-Z0-9_]+{re.escape(pattern)}(?![a-zA-Z0-9_])'
            )
    return _LEX_CACHE[key]


def count_lexical_matches(
    code: str,
    selected_lexical: list[tuple[str, str]],
) -> Counter:
    """
    Count occurrences of each selected lexical pattern (prefix/suffix) in code.
    Returns {pattern_str: count}.
    """
    counts: Counter = Counter()
    for pattern, pat_type in selected_lexical:
        regex = _get_lexical_regex(pattern, pat_type)
        matches = regex.findall(code)
        if matches:
            counts[pattern] = len(matches)
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate selection
# ═══════════════════════════════════════════════════════════════════════════════

def build_candidate_pool(
    mining_results: dict,
    tokenizer,
    tok_type: str,
) -> list[dict]:
    """
    Merge AST + lexical patterns into one ranked pool.

    Each candidate carries both the old heuristic score (total_savings_est)
    and the MDL net benefit (Delta_k = spi*freq - codebook_cost).
    The list is sorted by mdl_net_benefit descending.
    """
    candidates = []

    for skeleton, freq in mining_results.get("ast_patterns", {}).items():
        spi = savings_for_ast_skeleton(skeleton, tokenizer, tok_type)
        if spi > 0:
            cb = compute_codebook_cost(skeleton, tokenizer, tok_type)
            candidates.append({
                "type": "syntax",
                "pattern": skeleton,
                "frequency": freq,
                "savings_per_instance": spi,
                "total_savings_est": spi * freq,
                "codebook_cost": cb,
                "mdl_net_benefit": spi * freq - cb,
            })

    for prefix, freq in mining_results.get("lexical_prefixes", {}).items():
        spi = savings_for_lexical_pattern(prefix, tokenizer, tok_type)
        if spi > 0:
            cb = compute_codebook_cost(prefix, tokenizer, tok_type)
            candidates.append({
                "type": "prefix",
                "pattern": prefix,
                "frequency": freq,
                "savings_per_instance": spi,
                "total_savings_est": spi * freq,
                "codebook_cost": cb,
                "mdl_net_benefit": spi * freq - cb,
            })

    for suffix, freq in mining_results.get("lexical_suffixes", {}).items():
        spi = savings_for_lexical_pattern(suffix, tokenizer, tok_type)
        if spi > 0:
            cb = compute_codebook_cost(suffix, tokenizer, tok_type)
            candidates.append({
                "type": "suffix",
                "pattern": suffix,
                "frequency": freq,
                "savings_per_instance": spi,
                "total_savings_est": spi * freq,
                "codebook_cost": cb,
                "mdl_net_benefit": spi * freq - cb,
            })

    candidates.sort(key=lambda c: c["mdl_net_benefit"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_budget(
    tok_key: str,
    tokenizer,
    tok_type: str,
    candidates: list[dict],
    samples: list[str],
    budget: int,
    baseline_tokens_list: list[int],
    ast_skeletons_list: list[Counter],
    baseline_token_counts: Counter,
    total_original_bytes: int,
) -> dict:
    """
    Select Top-K operators and measure compression on eval samples.

    Returns compression metrics AND information-theoretic metrics:
    - MDL total description length (bits)
    - Token-level entropy before and after compression
    - Bits-per-byte before and after compression
    """
    selected = candidates[:budget]

    selected_skeletons = {
        c["pattern"] for c in selected if c["type"] == "syntax"
    }
    selected_lexical = [
        (c["pattern"], c["type"])
        for c in selected if c["type"] in ("prefix", "suffix")
    ]
    savings_map = {c["pattern"]: c["savings_per_instance"] for c in selected}

    n_syntax = len(selected_skeletons)
    n_lexical = len(selected_lexical)

    total_codebook_cost = sum(c["codebook_cost"] for c in selected)

    baseline_total = 0
    savings_total = 0
    ast_savings_total = 0
    lex_savings_total = 0

    for idx, code in enumerate(samples):
        baseline_total += baseline_tokens_list[idx]

        ast_counts = ast_skeletons_list[idx]
        ast_sav = sum(
            savings_map[sk] * cnt
            for sk, cnt in ast_counts.items()
            if sk in selected_skeletons
        )

        lex_counts = count_lexical_matches(code, selected_lexical)
        lex_sav = sum(savings_map[p] * cnt for p, cnt in lex_counts.items())

        savings_total += ast_sav + lex_sav
        ast_savings_total += ast_sav
        lex_savings_total += lex_sav

    compressed_total = max(1, baseline_total - savings_total)
    ratio = compressed_total / baseline_total if baseline_total else 1.0

    V0 = get_vocab_size(tokenizer, tok_type)
    V_aug = V0 + len(selected)

    mdl_score = compute_mdl_score(compressed_total, V_aug,
                                  total_codebook_cost, V0)
    baseline_mdl = baseline_total * math.log2(V0)

    baseline_entropy = compute_token_entropy(baseline_token_counts)
    baseline_bpb = compute_bits_per_byte(baseline_total, V0, total_original_bytes)
    compressed_bpb = compute_bits_per_byte(compressed_total, V_aug, total_original_bytes)

    return {
        "tokenizer": tok_key,
        "budget": budget,
        "baseline_tokens": baseline_total,
        "compressed_tokens": compressed_total,
        "ratio": ratio,
        "reduction_pct": (1 - ratio) * 100,
        "ast_savings": ast_savings_total,
        "lex_savings": lex_savings_total,
        "n_syntax_ops": n_syntax,
        "n_lexical_ops": n_lexical,
        "num_files": len(samples),
        "codebook_cost_tokens": total_codebook_cost,
        "mdl_score": mdl_score,
        "baseline_mdl": baseline_mdl,
        "mdl_reduction_pct": (1 - mdl_score / baseline_mdl) * 100 if baseline_mdl > 0 else 0,
        "baseline_entropy": baseline_entropy,
        "baseline_bpb": baseline_bpb,
        "compressed_bpb": compressed_bpb,
        "vocab_base": V0,
        "vocab_augmented": V_aug,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report & persistence
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: list[dict], mdl_optimal: dict[str, int]):
    print("\n" + "=" * 110)
    print("  OPERATOR-BASED COMPRESSION — MDL FRAMEWORK EVALUATION")
    print("=" * 110)

    header = (f"  {'Tokenizer':<18} {'Budget':>6} {'Baseline':>10} "
              f"{'Compressed':>10} {'Reduct%':>8} "
              f"{'MDL_L':>12} {'MDL_R%':>7} "
              f"{'BPB_base':>8} {'BPB_comp':>8} "
              f"{'CB_cost':>7}")
    print(header)
    print("-" * 110)

    for r in results:
        mdl_star = " *" if r["budget"] == mdl_optimal.get(r["tokenizer"]) else ""
        print(f"  {r['tokenizer']:<18} {r['budget']:>5}{mdl_star:1s} "
              f"{r['baseline_tokens']:>10,} {r['compressed_tokens']:>10,} "
              f"{r['reduction_pct']:>7.1f}% "
              f"{r['mdl_score']:>12,.0f} {r['mdl_reduction_pct']:>6.1f}% "
              f"{r['baseline_bpb']:>8.3f} {r['compressed_bpb']:>8.3f} "
              f"{r['codebook_cost_tokens']:>7}")

    print(f"\n  (* = MDL-optimal K*)")

    for tok_key, k_star in mdl_optimal.items():
        print(f"  MDL-optimal K* for {tok_key}: {k_star}")

    print(f"\n  ── Baseline token-level entropy ──")
    seen = set()
    for r in results:
        if r["tokenizer"] not in seen:
            seen.add(r["tokenizer"])
            print(f"  {r['tokenizer']:<18}  H = {r['baseline_entropy']:.4f} bits")

    print(f"\n  ── SimPy Reported (paper) ──")
    for tok_key, data in SIMPY_REPORTED.items():
        print(f"  {tok_key:<18}  reduction = {data['reduction_pct']:.1f}%")
    print("=" * 110)


def save_results(results: list[dict], candidates_by_tok: dict,
                 mdl_optimal: dict[str, int]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "tokenizer", "budget", "baseline_tokens", "compressed_tokens",
        "ratio", "reduction_pct", "ast_savings", "lex_savings",
        "n_syntax_ops", "n_lexical_ops", "num_files",
        "codebook_cost_tokens", "mdl_score", "baseline_mdl", "mdl_reduction_pct",
        "baseline_entropy", "baseline_bpb", "compressed_bpb",
        "vocab_base", "vocab_augmented",
    ]
    csv_path = RESULTS_DIR / "compression_report_mdl.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in ("ratio", "reduction_pct", "mdl_reduction_pct",
                       "baseline_entropy", "baseline_bpb", "compressed_bpb"):
                if k in row:
                    row[k] = f"{row[k]:.6f}"
            row["mdl_score"] = f"{r['mdl_score']:.2f}"
            row["baseline_mdl"] = f"{r['baseline_mdl']:.2f}"
            writer.writerow(row)
    print(f"\n[eval] CSV saved to {csv_path}")

    detail = {
        "mdl_optimal_k": mdl_optimal,
        "results": results,
        "top_operators_per_tokenizer": {},
    }
    for tok_key, cands in candidates_by_tok.items():
        detail["top_operators_per_tokenizer"][tok_key] = [
            {
                "type": c["type"],
                "pattern": c["pattern"],
                "savings_per_instance": c["savings_per_instance"],
                "frequency": c["frequency"],
                "total_savings_est": c["total_savings_est"],
                "codebook_cost": c["codebook_cost"],
                "mdl_net_benefit": c["mdl_net_benefit"],
                "marginal_delta_bits": c.get("marginal_delta_bits"),
            }
            for c in cands[:100]
        ]

    json_path = RESULTS_DIR / "eval_detail_mdl.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"[eval] Detail JSON saved to {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    budgets: Optional[list[int]] = None,
    tokenizer_keys: Optional[list[str]] = None,
) -> list[dict]:
    if budgets is None:
        budgets = OPERATOR_BUDGETS
    if tokenizer_keys is None:
        tokenizer_keys = list(EVAL_TOKENIZERS.keys())

    mining_path = CACHE_DIR / "mining_results.json"
    if not mining_path.exists():
        raise FileNotFoundError(
            f"Mining results not found: {mining_path}. Run frequency_miner.py first."
        )
    with open(mining_path, "r", encoding="utf-8") as f:
        mining = json.load(f)

    n_ast = len(mining.get("ast_patterns", {}))
    n_pre = len(mining.get("lexical_prefixes", {}))
    n_suf = len(mining.get("lexical_suffixes", {}))
    print(f"[eval] Loaded mining results: {n_ast} AST patterns, "
          f"{n_pre} prefixes, {n_suf} suffixes")

    print("[eval] Loading evaluation samples ...")
    samples = load_eval_samples()
    print(f"[eval] {len(samples)} evaluation samples loaded")

    total_original_bytes = sum(len(s.encode("utf-8")) for s in samples)
    print(f"[eval] Total original bytes: {total_original_bytes:,}")

    all_results = []
    candidates_by_tok = {}
    mdl_optimal = {}

    for tok_key in tokenizer_keys:
        if tok_key not in EVAL_TOKENIZERS:
            print(f"[eval] Unknown tokenizer key: {tok_key}, skipping")
            continue

        cfg = EVAL_TOKENIZERS[tok_key]
        print(f"\n{'='*60}")
        print(f"  TOKENIZER: {tok_key}")
        print(f"{'='*60}")

        tokenizer, tok_type = load_tokenizer(tok_key, cfg)
        V0 = get_vocab_size(tokenizer, tok_type)
        candidates = build_candidate_pool(mining, tokenizer, tok_type)
        candidates_by_tok[tok_key] = candidates

        print(f"  Candidates with positive token savings: {len(candidates)}")
        if candidates:
            top3 = candidates[:3]
            for c in top3:
                print(f"    [{c['type']}] \"{c['pattern'][:50]}\" "
                      f"spi={c['savings_per_instance']} freq={c['frequency']} "
                      f"net_benefit={c['mdl_net_benefit']:,}")

        print(f"  Pre-computing baselines & AST skeletons ...")
        baseline_tokens_list: list[int] = []
        ast_skeletons_list: list[Counter] = []
        baseline_token_counts: Counter = Counter()

        for code in tqdm(samples, desc=f"  [{tok_key}] precompute", leave=False):
            tokens = encode(tokenizer, tok_type, code)
            baseline_tokens_list.append(len(tokens))
            baseline_token_counts.update(tokens)
            ast_skeletons_list.append(get_all_ast_skeletons(code))

        N_baseline = sum(baseline_tokens_list)

        print(f"  Computing eval-corpus frequencies for MDL selection ...")
        eval_freq = compute_eval_frequencies(candidates, samples, ast_skeletons_list)
        n_eval_active = sum(1 for c in candidates if eval_freq.get(c["pattern"], 0) > 0)
        print(f"  Candidates with eval-corpus matches: {n_eval_active}/{len(candidates)}")

        k_star = greedy_mdl_select(candidates, N_baseline, V0, eval_freq=eval_freq)
        mdl_optimal[tok_key] = k_star

        print(f"  V0 = {V0:,}  |  N_baseline = {N_baseline:,}")
        print(f"  MDL-optimal K* = {k_star}  (greedy, eval-freq, vocab expansion)")
        if k_star > 0 and k_star <= len(candidates):
            last_accepted = candidates[k_star - 1]
            print(f"    Last accepted:  DeltaL = {last_accepted['marginal_delta_bits']:,.0f} bits")
        if k_star < len(candidates):
            first_rejected = candidates[k_star]
            print(f"    First rejected: DeltaL = {first_rejected.get('marginal_delta_bits', float('inf')):,.0f} bits")

        eval_budgets = sorted(set(budgets) | {k_star}) if k_star > 0 else budgets
        tok_results = []
        for budget in eval_budgets:
            if budget <= 0:
                continue
            result = evaluate_budget(
                tok_key, tokenizer, tok_type, candidates, samples, budget,
                baseline_tokens_list, ast_skeletons_list,
                baseline_token_counts, total_original_bytes,
            )
            all_results.append(result)
            tok_results.append(result)
            print(f"  budget={budget:>4}{'*' if budget == k_star else ' '}: "
                  f"reduction={result['reduction_pct']:.1f}%  "
                  f"MDL_reduction={result['mdl_reduction_pct']:.1f}%  "
                  f"bpb={result['compressed_bpb']:.3f}  "
                  f"(AST={result['ast_savings']:,}  LEX={result['lex_savings']:,})")

        best_mdl_result = min(tok_results, key=lambda r: r["mdl_score"])
        print(f"  >> L_total minimum at budget={best_mdl_result['budget']} "
              f"(MDL_R={best_mdl_result['mdl_reduction_pct']:.1f}%)")

    all_results.sort(key=lambda r: (r["tokenizer"], r["budget"]))
    print_report(all_results, mdl_optimal)
    save_results(all_results, candidates_by_tok, mdl_optimal)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--tokenizers", nargs="+", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(budgets=args.budgets, tokenizer_keys=args.tokenizers)
