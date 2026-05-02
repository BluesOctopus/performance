"""v2_eval rows include Plan A fields."""

from __future__ import annotations

from eval.v2_eval import EvalResult, evaluate
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
    assert hasattr(r, "stage3_selected_units")
    fields = [f.name for f in EvalResult.__dataclass_fields__.values()]
    assert "stage3_dictionary_coverage" in fields
    assert r.effective_total_tokens == r.sequence_final_tokens + r.final_vocab_intro_tokens
    assert r.final_vocab_intro_tokens == (
        r.stage1_vocab_intro_tokens
        + r.stage2_vocab_intro_tokens
        + r.stage3_vocab_intro_tokens
    )
