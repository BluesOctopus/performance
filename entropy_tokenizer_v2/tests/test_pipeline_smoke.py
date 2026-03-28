from types import SimpleNamespace

import pipeline
from tests.fixtures import SimpleOfflineTokenizer


def test_pipeline_stage_order_and_output(monkeypatch) -> None:
    calls: list[str] = []

    def fake_stage1_with_stats(source, repo_config, tokenizer, tok_type):
        calls.append("stage1")
        assert source == "x = 1"
        return "<SYN_0> x\nx = 1", {}

    def fake_stage2(text, *, profile, mode):
        calls.append("stage2")
        assert profile == "stage2_parseable"
        assert mode == "linewise"
        return text + "\n#cleaned"

    def fake_stage3(text, repo_config):
        calls.append("stage3")
        return text.replace("x", "<VAR>")

    monkeypatch.setattr(pipeline, "apply_stage1_with_stats", fake_stage1_with_stats)
    monkeypatch.setattr(pipeline, "apply_stage2", fake_stage2)
    monkeypatch.setattr(pipeline, "apply_stage3", fake_stage3)

    repo_config = SimpleNamespace(
        replacement_map={"x": "<VAR>"},
        skeleton_candidates=lambda: [],
    )
    out, breakdown = pipeline.apply_pipeline(
        "x = 1",
        repo_config,
        tokenizer=None,
        tok_type="hf",
        count_fn=len,
        stage2_profile="stage2_parseable",
        stage2_mode="linewise",
    )

    assert calls == ["stage1", "stage2", "stage3"]
    assert out == "<SYN_0> <VAR>\n<VAR> = 1\n#cleaned"
    assert breakdown.baseline_tokens == len("x = 1")
    assert breakdown.after_syntax == len("<SYN_0> x\nx = 1")
    assert breakdown.after_cleaning == len("<SYN_0> x\nx = 1\n#cleaned")
    assert breakdown.after_replacement == len("<SYN_0> <VAR>\n<VAR> = 1\n#cleaned")


def test_pipeline_smoke_offline_deterministic() -> None:
    class MiniRepoConfig:
        replacement_map = {"name": "<VAR>"}

        @staticmethod
        def skeleton_candidates():
            return []

    source = "name = 1\nprint(name)"
    tokenizer = SimpleOfflineTokenizer()
    out1, breakdown1 = pipeline.apply_pipeline(
        source,
        MiniRepoConfig(),
        tokenizer=tokenizer,
        tok_type="hf",
    )
    out2, breakdown2 = pipeline.apply_pipeline(
        source,
        MiniRepoConfig(),
        tokenizer=tokenizer,
        tok_type="hf",
    )

    assert isinstance(out1, str)
    assert isinstance(breakdown1.baseline_tokens, int)
    assert out1 == out2
    assert breakdown1 == breakdown2
