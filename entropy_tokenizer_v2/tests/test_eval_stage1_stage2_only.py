"""Offline tests for Stage1+Stage2-only eval script (no HF, no Stage3)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_stage1_stage2():
    name = "eval_stage1_stage2_only_under_test"
    if name in sys.modules:
        return sys.modules[name]
    p = ROOT / "scripts" / "eval_stage1_stage2_only.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_safe_parse_success() -> None:
    ev = _load_eval_stage1_stage2()
    assert ev.safe_parse_success("def f():\n    pass\n") is True
    assert ev.safe_parse_success("def bad(\n") is False


def test_run_experiment_adapted_safe_no_stage3(monkeypatch: pytest.MonkeyPatch) -> None:
    ev = _load_eval_stage1_stage2()
    calls: list[str] = []

    def boom(*_a, **_k):
        calls.append("stage3")
        raise AssertionError("Stage3 must not be called")

    monkeypatch.setattr("pipeline.apply_stage3", boom)

    samples = [
        {"path": "src/a.py", "text": "def f():\n    return 1\n", "base_tokens": 0},
        {"path": "src/b.py", "text": "def g():\n    return 2\n", "base_tokens": 0},
    ]
    from lossy_cleaner import lossless_clean

    clean_texts = [lossless_clean(s["text"])[0] for s in samples]

    class Tok:
        def encode(self, text, allowed_special="all"):
            return list(text.encode("utf-8"))

        @property
        def n_vocab(self):
            return 50000

    from stage2.config import STAGE2_ADAPTED_ORDER_LABEL

    ctx = ev.build_stage1_mining_context(clean_texts, Tok(), "tiktoken")
    summary, rows = ev.run_experiment_batch(
        samples,
        experiment_name=ev.EXPERIMENT_ADAPTED_SAFE,
        tokenizer=Tok(),
        tok_type="tiktoken",
        ctx=ctx,
        stage2_profile="safe",
        stage2_order=STAGE2_ADAPTED_ORDER_LABEL,
        apply_vocab_amort=True,
    )
    assert STAGE2_ADAPTED_ORDER_LABEL in summary.get("stage2_order", "")
    assert calls == []
    assert len(rows) == 2
    assert summary["experiment_name"] == ev.EXPERIMENT_ADAPTED_SAFE


def test_run_experiment_adapted_agg_indent_post(monkeypatch: pytest.MonkeyPatch) -> None:
    ev = _load_eval_stage1_stage2()
    monkeypatch.setattr("pipeline.apply_stage3", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError))

    samples = [
        {
            "path": "mod/x.py",
            "text": "def f():\n    if True:\n        return 3\n",
            "base_tokens": 0,
        },
    ]
    from lossy_cleaner import lossless_clean

    clean, _ = lossless_clean(samples[0]["text"])

    class Tok:
        def encode(self, text, allowed_special="all"):
            return list(text.encode("utf-8"))

        @property
        def n_vocab(self):
            return 50000

    ctx = ev.build_stage1_mining_context([clean], Tok(), "tiktoken")
    summary, rows = ev.run_experiment_batch(
        samples,
        experiment_name=ev.EXPERIMENT_ADAPTED_AGG,
        tokenizer=Tok(),
        tok_type="tiktoken",
        ctx=ctx,
        stage2_profile="aggressive_upper_bound",
        stage2_order="pre_safe -> stage1 -> post_surface",
        apply_vocab_amort=True,
    )
    assert summary["stage2_post_total_removed_indent_chars"] > 0
    assert rows[0]["stage2_post_removed_indent_chars"] > 0


def test_process_one_file_pre_path_aware_docstring() -> None:
    ev = _load_eval_stage1_stage2()
    from lossy_cleaner import lossless_clean

    sample = {
        "path": "tests/test_utils.py",
        "text": 'def _h():\n    """x"""\n    return 1\n',
        "base_tokens": 0,
    }

    class Tok:
        def encode(self, text, allowed_special="all"):
            return list(text.encode("utf-8"))

        @property
        def n_vocab(self):
            return 50000

    clean, _ = lossless_clean(sample["text"])
    ctx = ev.build_stage1_mining_context([clean], Tok(), "tiktoken")
    r = ev.process_one_file_experiment(
        sample,
        tokenizer=Tok(),
        tok_type="tiktoken",
        stage1_repo=ctx["repo_config"],
        experiment_name=ev.EXPERIMENT_ADAPTED_SAFE,
    )
    assert "tests" in r["stage2_pre_docstring_path_context_labels"]


def test_write_stage1_stage2_only_reports(tmp_path: Path) -> None:
    ev = _load_eval_stage1_stage2()
    samples = ev.offline_regression_samples()[:2]

    class Tok:
        def encode(self, text, allowed_special="all"):
            return list(text.encode("utf-8"))

        @property
        def n_vocab(self):
            return 50000

    summaries, per_file, ctx, rule_rows, adaptation_examples, layout_sidecar = (
        ev.run_all_experiments(samples, tokenizer=Tok(), tok_type="tiktoken")
    )
    manifest = {
        "dataset_name": "unit",
        "token_budget": 1,
        "actual_accumulated_base_tokens": 1,
        "num_files": len(samples),
    }
    ev.write_stage1_stage2_only_reports(
        summaries,
        per_file,
        ctx=ctx,
        manifest=manifest,
        output_dir=tmp_path,
        tokenizer_name="unit",
        resume_used=False,
        run_status="test",
        rule_breakdown_rows=rule_rows,
        adaptation_examples=adaptation_examples,
        layout_sidecar=layout_sidecar,
    )
    assert (tmp_path / "stage1_stage2_only_summary.csv").exists()
    assert (tmp_path / "stage1_stage2_only_summary.json").exists()
    assert (tmp_path / "stage1_stage2_only_per_file.csv").exists()
    assert (tmp_path / "stage1_stage2_only_selected_skeletons.csv").exists()
    assert (tmp_path / "stage1_stage2_only_vocab_tokens.json").exists()
    assert (tmp_path / "stage1_stage2_only_manifest.json").exists()
    assert (tmp_path / "stage2_rule_breakdown.csv").exists()
    assert (tmp_path / "stage1_stage2_adaptation_examples.jsonl").exists()
    assert (tmp_path / "stage1_stage2_adaptation_examples.md").exists()
    assert (tmp_path / "stage2_layout_encoding_summary.csv").exists()
    assert (tmp_path / "stage2_layout_encoding_examples.md").exists()
    assert (tmp_path / "stage2_layout_encoding_examples.jsonl").exists()
    mj = json.loads((tmp_path / "stage1_stage2_only_manifest.json").read_text(encoding="utf-8"))
    assert "experiments" in mj
    assert ev.EXPERIMENT_LAYOUT_EXPERIMENTAL in mj["experiments"]
