"""
Global configuration for the Operator-Based Hierarchical Tokenizer project.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"

# StarCoderData access
HF_DATASET_REPO = "bigcode/starcoderdata"
HF_DATASET_LANG = "python"
HF_TOKEN = "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"

# Sampling config
SAMPLE_MIN_STARS = 0
SAMPLE_MAX_BYTES = 100 * 1024 * 1024       # ~100 MB
SAMPLE_MAX_BYTES_QUICK = 10 * 1024 * 1024   # ~10 MB for quick mode
SAMPLE_SEED = 42
SAMPLE_SHUFFLE_BUFFER = 5000

# Base model tokenizer
BASE_TOKENIZER = "bigcode/starcoder2-3b"

# ─── Operator mining config ──────────────────────────────────────────────────

AST_MIN_FREQ = 50           # minimum occurrences for an AST skeleton to be a candidate
LEXICAL_MIN_FREQ = 100      # minimum occurrences for a lexical pattern to be a candidate

# Budget levels for ablation (number of operators to select)
OPERATOR_BUDGETS = [20, 50, 100, 200, 300, 400, 500, 700, 1000, 1500]

# ─── MDL (Minimum Description Length) framework ──────────────────────────────

CODEBOOK_OVERHEAD_PER_OP = 1   # fixed token overhead per operator (the op token itself)

# ─── Multi-tokenizer evaluation (aligned with SimPy) ─────────────────────────

EVAL_TOKENIZERS = {
    "gpt4": {
        "type": "tiktoken",
        "tiktoken_model": "gpt-4",
    },
    "codegen-350M-mono": {
        "type": "hf",
        "name": "Salesforce/codegen-350M-mono",
    },
    "santacoder": {
        "type": "hf",
        "name": "bigcode/santacoder",
    },
}

# starcoderdata_100star evaluation
EVAL_100STAR_NUM_SAMPLES = 1000
EVAL_100STAR_DATASET = "zhensuuu/starcoderdata_100star_py"

# ─── SimPy reported results (from paper, for comparison) ─────────────────────

SIMPY_REPORTED = {
    "gpt4":              {"reduction_pct": 10.4},
    "codegen-350M-mono": {"reduction_pct": 13.5},
    "santacoder":        {"reduction_pct": 8.8},
}
