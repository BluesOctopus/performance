"""Hybrid AB backend smoke and accounting checks."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EVAL = _ROOT / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from config import EVAL_TOKENIZERS
from repo_miner import mine_from_sources
from v2_eval import evaluate


def test_hybrid_ab_eval_fields_and_accounting():
    sources = [
        "user_name_long_identifier = 1\nprint(user_name_long_identifier)\n",
        "msg = 'Please verify the user login request before proceeding now'\n",
        "msg2 = 'Please verify user login request before proceeding immediately'\n",
    ]
    rc = mine_from_sources(
        sources,
        tokenizer_key="gpt4",
        tokenizer_cfg=EVAL_TOKENIZERS["gpt4"],
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend="hybrid_ab",
    )
    r = evaluate(sources, rc, "gpt4", EVAL_TOKENIZERS["gpt4"])
    assert r.stage3_backend == "hybrid_ab"
    assert r.stage3_ab_mode == "exact_only"
    assert r.stage3_ab_similarity_kind == "lexical_bow_cosine"
    assert r.final_vocab_intro_tokens == (
        r.stage1_vocab_intro_tokens + r.stage2_vocab_intro_tokens + r.stage3_vocab_intro_tokens
    )
    assert r.effective_total_tokens == r.sequence_final_tokens + r.final_vocab_intro_tokens
    assert r.stage3_vocab_intro_tokens == r.stage3_ab_a_intro_tokens + r.stage3_ab_b_intro_tokens
    assert r.stage3_component_saved == r.stage3_ab_a_sequence_saved + r.stage3_ab_b_sequence_saved
    assert r.stage3_selected_units >= r.stage3_selected_units_exact
    assert r.stage3_used_units_exact >= 0


def test_legacy_and_plan_a_still_work():
    sources = ["x = 1\nprint(x)\n", "y = 'config_key_value'\n"]
    for backend in ("legacy", "plan_a"):
        rc = mine_from_sources(
            sources,
            tokenizer_key="gpt4",
            tokenizer_cfg=EVAL_TOKENIZERS["gpt4"],
            cache=False,
            verbose=False,
            min_freq=1,
            stage3_backend=backend,
        )
        r = evaluate(sources, rc, "gpt4", EVAL_TOKENIZERS["gpt4"])
        assert r.stage3_backend == backend
