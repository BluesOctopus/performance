"""RepoConfig Plan A mining + JSON roundtrip."""

from __future__ import annotations

from repo_miner import RepoConfig, mine_from_sources, load_plan_a_codebooks


def test_repo_config_plan_a_json_roundtrip():
    cfg = RepoConfig(
        stage3_backend="plan_a",
        stage3_escape_prefix="__L__",
        stage3_codebook_version="v1",
        stage3_plan_a_codebooks={
            "variable": {
                "field_name": "variable",
                "version": "v1",
                "escape_prefix": "__L__",
                "metadata": {},
                "assignments": [
                    {
                        "literal": "foo",
                        "code": "a",
                        "raw_token_cost": 2,
                        "code_token_cost": 1,
                        "expected_gain": 0.5,
                    }
                ],
            }
        },
        stage3_plan_a_report={"summary": {"x": 1}},
        stage3_plan_a_summary={"stage3_plan_a_assignments_count": 1},
    )
    cfg2 = RepoConfig.from_json(cfg.to_json())
    assert cfg2.stage3_backend == "plan_a"
    assert "variable" in cfg2.stage3_plan_a_codebooks
    books = load_plan_a_codebooks(cfg2)
    assert books["variable"].assignments[0].literal == "foo"


def test_mine_small_corpus_plan_a():
    sources = [
        "def f():\n    serve_auth = 1\n    return serve_auth\n",
        "serve_auth = serve_auth + 1\n",
    ]
    from config import EVAL_TOKENIZERS

    cfg = EVAL_TOKENIZERS["gpt4"]
    rc = mine_from_sources(
        sources,
        tokenizer_key="gpt4",
        tokenizer_cfg=cfg,
        cache_name=None,
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend="plan_a",
    )
    assert rc.stage3_backend == "plan_a"
    assert isinstance(rc.stage3_plan_a_codebooks, dict)
    books = load_plan_a_codebooks(rc)
    assert isinstance(books, dict)
