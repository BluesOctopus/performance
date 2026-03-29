"""Paths, tokenizer presets, and hyperparameters for Stages 1–3."""

import copy
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

# Stage3 backend: legacy replacement_map vs Plan A literal codec vs hybrid AB routing.
STAGE3_BACKEND = os.getenv("ET_STAGE3_BACKEND", "legacy").strip().lower()
if STAGE3_BACKEND not in {"legacy", "plan_a", "hybrid_ab"}:
    STAGE3_BACKEND = "legacy"

STAGE3_CODEBOOK_VERSION = os.getenv("ET_STAGE3_CODEBOOK_VERSION", "v1")
STAGE3_ESCAPE_PREFIX = os.getenv("ET_STAGE3_ESCAPE_PREFIX", "__L__")
STAGE3_PLAN_A_ENABLED_CATEGORIES = tuple(
    x.strip()
    for x in os.getenv(
        "ET_STAGE3_PLAN_A_ENABLED_CATEGORIES", "variable,attribute,string"
    ).split(",")
    if x.strip()
)
def _parse_plan_a_max_assignments() -> dict[str, int]:
    raw = os.getenv(
        "ET_STAGE3_PLAN_A_MAX_ASSIGNMENTS_PER_FIELD",
        "variable:256,attribute:128,string:128",
    )
    out: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


STAGE3_PLAN_A_MIN_GAIN = float(os.getenv("ET_STAGE3_PLAN_A_MIN_GAIN", "0.001"))
STAGE3_PLAN_A_MAX_ASSIGNMENTS_PER_FIELD = _parse_plan_a_max_assignments()
STAGE3_PLAN_A_USE_TIKTOKEN = os.getenv("ET_STAGE3_PLAN_A_USE_TIKTOKEN", "0").lower() in (
    "1",
    "true",
    "yes",
)
STAGE3_PLAN_A_VOCAB_SCOPE = os.getenv("ET_STAGE3_PLAN_A_VOCAB_SCOPE", "used_only")
STAGE3_PLAN_A_COST_MODEL = os.getenv("ET_STAGE3_PLAN_A_COST_MODEL", "real_surface_form")
STAGE3_ARTIFACT_DIR = Path(
    os.getenv("ET_STAGE3_ARTIFACT_DIR", str(RESULTS_DIR / "stage3_plan_a"))
)

# Plan A tokenizer-aware profiles (see docs/stage3_plan_a_integration.md).
# Override with ET_STAGE3_PLAN_A_PROFILE=default|gpt4_conservative|gpt4_va_only
STAGE3_PLAN_A_POST_PRUNE = os.getenv("ET_STAGE3_PLAN_A_POST_PRUNE", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Stage3 Hybrid AB knobs (A=exact aliasing, B=semantic clustering).
STAGE3_AB_FREE_TEXT_MIN_CHARS = int(os.getenv("ET_STAGE3_AB_FREE_TEXT_MIN_CHARS", "24"))
STAGE3_AB_FREE_TEXT_MIN_WORDS = int(os.getenv("ET_STAGE3_AB_FREE_TEXT_MIN_WORDS", "4"))
STAGE3_AB_B_SIMILARITY_THRESHOLD = float(
    os.getenv("ET_STAGE3_AB_B_SIMILARITY_THRESHOLD", "0.82")
)
STAGE3_AB_B_RISK_THRESHOLD = float(os.getenv("ET_STAGE3_AB_B_RISK_THRESHOLD", "0.72"))
STAGE3_AB_B_MIN_CLUSTER_SIZE = int(os.getenv("ET_STAGE3_AB_B_MIN_CLUSTER_SIZE", "2"))
STAGE3_AB_ENABLE_B = os.getenv("ET_STAGE3_AB_ENABLE_B", "1").lower() in ("1", "true", "yes")


def resolve_hybrid_ab_settings(tokenizer_key: str) -> dict:
    """
    Resolve hybrid_ab runtime settings with tokenizer-aware defaults.

    gpt4 defaults to slightly stricter semantic merge threshold.
    """
    tok = (tokenizer_key or "").strip().lower()
    sim_default = 0.84 if tok == "gpt4" else STAGE3_AB_B_SIMILARITY_THRESHOLD
    risk_default = 0.74 if tok == "gpt4" else STAGE3_AB_B_RISK_THRESHOLD
    return {
        "free_text_min_chars": STAGE3_AB_FREE_TEXT_MIN_CHARS,
        "free_text_min_words": STAGE3_AB_FREE_TEXT_MIN_WORDS,
        "b_similarity_threshold": float(
            os.getenv("ET_STAGE3_AB_B_SIMILARITY_THRESHOLD", str(sim_default))
        ),
        "b_risk_threshold": float(os.getenv("ET_STAGE3_AB_B_RISK_THRESHOLD", str(risk_default))),
        "b_min_cluster_size": int(
            os.getenv("ET_STAGE3_AB_B_MIN_CLUSTER_SIZE", str(STAGE3_AB_B_MIN_CLUSTER_SIZE))
        ),
        "enable_b": os.getenv("ET_STAGE3_AB_ENABLE_B", "1" if STAGE3_AB_ENABLE_B else "0")
        .lower()
        in ("1", "true", "yes"),
    }


def plan_a_profile_name_for_tokenizer(tokenizer_key: str) -> str:
    """Default profile: gpt4 -> conservative; others -> default (gpt2-friendly)."""
    explicit = os.getenv("ET_STAGE3_PLAN_A_PROFILE", "").strip()
    if explicit:
        return explicit
    if tokenizer_key.strip().lower() == "gpt4":
        return os.getenv("ET_STAGE3_PLAN_A_PROFILE_GPT4_DEFAULT", "gpt4_conservative")
    return "default"


def resolve_plan_a_settings(tokenizer_key: str) -> dict:
    """
    Resolved Plan A knobs for this tokenizer (and env overrides).

    Returns dict with:
    profile_name, enabled_categories, min_gain, max_assignments_by_field,
    string_filter (dict or None), post_prune_enabled
    """
    profile = plan_a_profile_name_for_tokenizer(tokenizer_key)
    # Env overrides (highest priority for categories / gain / caps)
    cat_env = os.getenv("ET_STAGE3_PLAN_A_ENABLED_CATEGORIES", "").strip()
    mg_env = os.getenv("ET_STAGE3_PLAN_A_MIN_GAIN", "").strip()
    max_env = os.getenv("ET_STAGE3_PLAN_A_MAX_ASSIGNMENTS_PER_FIELD", "").strip()

    profiles: dict[str, dict] = {
        "default": {
            "enabled_categories": ("variable", "attribute", "string"),
            "min_gain": 0.001,
            "max_assignments": {"variable": 256, "attribute": 128, "string": 128},
            "string_filter": {"min_count": 1, "min_raw_token_cost": 0, "strict_heuristics": False},
            "post_prune": STAGE3_PLAN_A_POST_PRUNE,
        },
        "gpt4_conservative": {
            "enabled_categories": ("variable", "attribute", "string"),
            "min_gain": 0.003,
            "max_assignments": {"variable": 64, "attribute": 32, "string": 16},
            "string_filter": {"min_count": 2, "min_raw_token_cost": 8, "strict_heuristics": True},
            "post_prune": True,
        },
        "gpt4_va_only": {
            "enabled_categories": ("variable", "attribute"),
            "min_gain": 0.003,
            "max_assignments": {"variable": 64, "attribute": 32},
            "string_filter": None,
            "post_prune": True,
        },
    }
    base = copy.deepcopy(profiles.get(profile, profiles["default"]))
    if cat_env:
        base["enabled_categories"] = tuple(
            x.strip() for x in cat_env.split(",") if x.strip()
        )
    if mg_env:
        base["min_gain"] = float(mg_env)
    if max_env:
        out_max: dict[str, int] = {}
        for part in max_env.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            k, v = part.split(":", 1)
            out_max[k.strip()] = int(v.strip())
        base["max_assignments"] = out_max
    base["profile_name"] = profile
    pp_env = os.getenv("ET_STAGE3_PLAN_A_POST_PRUNE", "").strip()
    if pp_env:
        base["post_prune"] = pp_env.lower() in ("1", "true", "yes")
    return base

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
    r"<NL[0-8]>",
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
