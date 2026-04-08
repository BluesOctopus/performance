"""Paths, tokenizer presets, and hyperparameters for Stages 1–3."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR  = PROJECT_ROOT / "results"
CACHE_DIR    = PROJECT_ROOT / "cache"
DATA_DIR     = PROJECT_ROOT.parent.parent / "data"

HF_TOKEN            = "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"
EVAL_DATASET        = "zhensuuu/starcoderdata_100star_py"
EVAL_NUM_SAMPLES    = 1000

EVAL_TOKENIZERS = {
    "gpt4": {
        "type": "tiktoken",
        "tiktoken_model": "gpt-4",
    },
    "gpt2": {
        "type": "hf",
        "name": "gpt2",
    },
    "codegen-350M-mono": {
        "type": "hf",
        "name": "Salesforce/codegen-350M-mono",
    },
    "santacoder": {
        "type": "hf",
        "name": "bigcode/santacoder",
    },
    "deepseek": {
        "type": "hf",
        "name": "deepseek-ai/deepseek-coder-6.7b-instruct", # 使用 deepseek-coder 的分词器
    },
}

AST_MIN_FREQ          = 5      # skeleton must appear ≥ N times to be a candidate
MDL_CODEBOOK_OVERHEAD = 2      # tokens needed to encode one operator in codebook

# (enabled, is_lossy, description)
CLEANING_RULES = {
    "remove_comments":            (True,  False, "R01 Remove # inline comments"),
    "remove_docstrings":          (True,  True,  "R05 Remove docstrings [LOSSY]"),
    "use_minimalist_indent":      (True,  True,  "R04 Implicit IND after ':', explicit <D> [LOSSY]"),
    "remove_blank_lines":         (True,  False, "R02 Remove empty lines"),
    "remove_trailing_whitespace": (True,  False, "R03 Remove trailing spaces/tabs"),
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
