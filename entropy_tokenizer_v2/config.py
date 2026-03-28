"""Paths, tokenizer presets, and hyperparameters for Stages 1–3."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR  = PROJECT_ROOT / "results"
CACHE_DIR    = PROJECT_ROOT / "cache"
DATA_DIR     = PROJECT_ROOT.parent.parent / "data"

HF_TOKEN            = "**"
EVAL_DATASET        = "bigcode/starcoderdata"
EVAL_NUM_SAMPLES    = 1000

EVAL_TOKENIZERS = {
    "codebert": {
        "type": "hf",
        "name": "microsoft/codebert-base",
    },
    "gpt2": {
        "type": "hf",
        "name": "gpt2",
    },
    "codellama": {
        "type": "hf",
        "name": "codellama/CodeLlama-7b-hf",
    },
    "deepseek-coder": {
        "type": "hf",
        "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
    },
    "codegen": {
        "type": "hf",
        "name": "Salesforce/codegen-350M-mono",
    },
    "codex": {
        "type": "hf",
        "name": "Salesforce/codegen-350M-mono",
    },
    "santacoder": {
        "type": "hf",
        "name": "bigcode/santacoder",
    },
    "gpt-3.5": {
        "type": "tiktoken",
        "tiktoken_model": "gpt-3.5-turbo",
    },
    "gpt4": {
        "type": "tiktoken",
        "tiktoken_model": "gpt-4",
    },
}

AST_MIN_FREQ          = 5      # skeleton must appear ≥ N times to be a candidate
MDL_CODEBOOK_OVERHEAD = 2      # tokens needed to encode one operator in codebook

# (enabled, is_lossy, description)
CLEANING_RULES = {
    "remove_comments":            (False, False, "R01 Remove # inline comments [default off]"),
    "remove_blank_lines":         (True,  False, "R02 Remove empty lines"),
    "remove_trailing_whitespace": (True,  False, "R03 Remove trailing spaces/tabs"),
    "remove_docstrings":          (False, True,  "R05 Remove docstrings [LOSSY, default off]"),
    "remove_indentation":         (True,  True,  "R04 Remove all indentation [LOSSY]"),
}

SCORE_EPSILON             = 0.01   # ε in Score(w) denominator
SCORE_THRESHOLD_PERCENTILE = 0.70  # replace top (1-0.70)=30% by score

PLACEHOLDERS = {
    "variable":  "<VAR>",
    "attribute": "<ATTR>",
    "string":    "<STR>",
    "fstring":   "<FSTR>",
    "number":    "<NUM>",
}
