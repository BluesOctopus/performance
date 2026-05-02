from __future__ import annotations

from config import (
    AST_MIN_FREQ,
    EVAL_TOKENIZERS,
    STAGE1_HYBRID_AB_AST_MIN_FREQ,
    STAGE1_HYBRID_AB_MIN_TOTAL_NET_SAVING,
    STAGE1_HYBRID_AB_SCORE_THRESHOLD_PERCENTILE,
)
from repo_miner import mine_from_sources


def test_hybrid_ab_mining_sets_stage1_override_flags() -> None:
    sources = ["def f():\n    return 1\n", "def g():\n    return 2\n"]
    rc = mine_from_sources(
        sources,
        tokenizer_key="gpt4",
        tokenizer_cfg=EVAL_TOKENIZERS["gpt4"],
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend="hybrid_ab",
    )
    assert rc.hybrid_ab_stage1_override_used is True
    assert rc.hybrid_ab_stage1_ast_min_freq_used == STAGE1_HYBRID_AB_AST_MIN_FREQ
    assert rc.hybrid_ab_stage1_min_total_net_saving_used == STAGE1_HYBRID_AB_MIN_TOTAL_NET_SAVING
    assert rc.hybrid_ab_stage1_score_percentile_config == STAGE1_HYBRID_AB_SCORE_THRESHOLD_PERCENTILE


def test_legacy_mining_does_not_set_hybrid_ab_stage1_override() -> None:
    sources = ["x = 1\n"]
    rc = mine_from_sources(
        sources,
        tokenizer_key="gpt4",
        tokenizer_cfg=EVAL_TOKENIZERS["gpt4"],
        cache=False,
        verbose=False,
        min_freq=AST_MIN_FREQ,
        stage3_backend="legacy",
    )
    assert rc.hybrid_ab_stage1_override_used is False
    assert rc.hybrid_ab_stage1_ast_min_freq_used is None
