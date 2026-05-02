from __future__ import annotations

from config import EVAL_TOKENIZERS
from pipeline import apply_pipeline
from repo_miner import _load_tokenizer, mine_from_sources


def test_repo_config_not_polluted_by_runtime_stage3_meta() -> None:
    sources = [
        "def _helper_name_for_aliasing():\n    return 1\nx = _helper_name_for_aliasing()\n",
        "msg = 'This is a long natural language sentence for lexical clustering baseline'\n",
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
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    _out, _bd = apply_pipeline(sources[0], rc, tok, tt)

    attrs = set(dir(rc))
    assert "_stage3_hybrid_last_meta" not in attrs
    assert not any(a.startswith("_runtime_") for a in attrs)
    assert not any(a.startswith("_last_") for a in attrs)

