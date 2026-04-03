"""Hybrid AB backend smoke and accounting checks."""

from __future__ import annotations

from config import EVAL_TOKENIZERS
from repo_miner import mine_from_sources
from eval.v2_eval import evaluate


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
    assert r.stage3_vocab_intro_tokens_raw >= r.stage3_vocab_intro_tokens_dedup
    assert r.effective_total_tokens_dedup <= r.effective_total_tokens
    assert r.stage3_component_saved == r.stage3_ab_a_sequence_saved + r.stage3_ab_b_sequence_saved
    assert r.stage3_selected_units >= r.stage3_selected_units_exact
    assert r.stage3_used_units_exact >= 0
    assert r.stage3_ab_a_route_reject_count >= 0
    assert r.stage3_ab_b_route_reject_count >= 0


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
