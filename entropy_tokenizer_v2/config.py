"""Paths, tokenizer presets, and hyperparameters for Stages 1–3."""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = Path(os.getenv("ET_DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR  = Path(os.getenv("ET_RESULTS_DIR", PROJECT_ROOT / "results"))
CACHE_DIR    = Path(os.getenv("ET_CACHE_DIR", PROJECT_ROOT / "cache"))

HF_TOKEN         = os.getenv("HF_TOKEN", "")
EVAL_DATASET     = os.getenv("ET_EVAL_DATASET", "zhensuuu/starcoderdata_100star_py")
EVAL_NUM_SAMPLES = int(os.getenv("ET_EVAL_NUM_SAMPLES", "1000"))

LOCAL_SAMPLE_FILE = Path(
    os.getenv("ET_LOCAL_SAMPLE_FILE", DATA_DIR / "starcoder_1m_tokens.txt")
)
HF_DISK_DATASET_FALLBACK = Path(
    os.getenv("ET_HF_DISK_DATASET_FALLBACK", DATA_DIR / "test")
)
SIMPY_DIR = Path(
    os.getenv("ET_SIMPY_DIR", PROJECT_ROOT / "third_party" / "Simpy-master")
)

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
}

AST_MIN_FREQ           = 20
MDL_CODEBOOK_OVERHEAD  = 2
STAGE1_MIN_OCCURRENCES = 2
STAGE1_MIN_TOTAL_NET_SAVING = 1
STAGE1_MIN_AVG_NET_SAVING = 0

# Placeholder regex fragments (consumed by ``placeholder_accounting.PLACEHOLDER_RE``).
PLACEHOLDER_PATTERNS = [
    r"<SYN_\d+>",
    r"<VAR>",
    r"<ATTR>",
    r"<STR>",
    r"<FSTR>",
    r"<NUM>",
]

# Vocab introduction cost for effective-total accounting.
VOCAB_COST_MODE = os.getenv("ET_VOCAB_COST_MODE", "serialized_definition")
VOCAB_COST_SCOPE = os.getenv("ET_VOCAB_COST_SCOPE", "corpus_once")
FIXED_VOCAB_TOKEN_COST = int(os.getenv("ET_FIXED_VOCAB_TOKEN_COST", "4"))

CLEANING_RULES = {
    "remove_comments":            (False, False, "R01 Remove # inline comments [default off]"),
    "remove_blank_lines":         (True,  False, "R02 Remove empty lines"),
    "remove_trailing_whitespace": (True,  False, "R03 Remove trailing spaces/tabs"),
    "remove_docstrings":          (False, True,  "R05 Remove docstrings [LOSSY, default off]"),
    "remove_indentation":         (True,  True,  "R04 Remove all indentation [LOSSY]"),
}

STAGE2_DEFAULT_PROFILE = os.getenv("ET_STAGE2_DEFAULT_PROFILE", "stage2_aggressive")
STAGE2_DEFAULT_MODE = os.getenv("ET_STAGE2_DEFAULT_MODE", "linewise")
STAGE2_PROFILE_FLAGS = {
    "stage2_parseable": {
        "remove_comments": False,
        "remove_blank_lines": True,
        "remove_trailing_whitespace": True,
        "remove_docstrings": False,
        "remove_indentation": False,
    },
    "stage2_aggressive": {
        "remove_comments": False,
        "remove_blank_lines": True,
        "remove_trailing_whitespace": True,
        "remove_docstrings": False,
        "remove_indentation": True,
    },
    # Stage1+Stage2 eval profiles (blockwise + docstrings; see scripts/eval_stage1_stage2_only.py).
    "stage2_safe": {
        "remove_comments": False,
        "remove_blank_lines": True,
        "remove_trailing_whitespace": True,
        "remove_docstrings": True,
        "remove_indentation": False,
    },
    "stage2_aggressive_upper_bound": {
        "remove_comments": True,
        "remove_blank_lines": True,
        "remove_trailing_whitespace": True,
        "remove_docstrings": True,
        "remove_indentation": True,
    },
}

SCORE_EPSILON              = 0.01
SCORE_THRESHOLD_PERCENTILE = 0.70

PLACEHOLDERS = {
    "variable":  "<VAR>",
    "attribute": "<ATTR>",
    "string":    "<STR>",
    "fstring":   "<FSTR>",
    "number":    "<NUM>",
}
