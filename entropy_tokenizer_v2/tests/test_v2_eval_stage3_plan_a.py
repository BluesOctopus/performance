"""v2_eval rows include Plan A fields."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EVAL = _ROOT / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from v2_eval import EvalResult, evaluate
from repo_miner import mine_from_sources
from config import EVAL_TOKENIZERS


def test_evaluate_plan_a_has_extra_columns():
    sources = ["serve_user = 1\nserve_user\n"]
    rc = mine_from_sources(
        sources,
        tokenizer_key="gpt4",
        tokenizer_cfg=EVAL_TOKENIZERS["gpt4"],
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend="plan_a",
    )
    tok_cfg = EVAL_TOKENIZERS["gpt4"]
    r = evaluate(sources, rc, "gpt4", tok_cfg)
    assert r.stage3_backend == "plan_a"
    assert hasattr(r, "stage3_vocab_intro_tokens")
    assert hasattr(r, "stage3_assignments")
    fields = [f.name for f in EvalResult.__dataclass_fields__.values()]
    assert "stage3_dictionary_coverage" in fields
