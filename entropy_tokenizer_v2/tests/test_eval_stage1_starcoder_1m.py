"""Offline tests for Stage1 StarCoder 1M eval helpers (no HF download in verify)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_script():
    name = "eval_stage1_starcoder_1m"
    if name in sys.modules:
        return sys.modules[name]
    p = ROOT / "scripts" / "eval_stage1_starcoder_1m.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_gpt4o_fallback_to_o200k_base() -> None:
    mock_enc = object()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("tiktoken.encoding_for_model", lambda *_: (_ for _ in ()).throw(KeyError("nope")))
        mp.setattr("tiktoken.get_encoding", lambda *_: mock_enc)
        from tokenizer_utils import resolve_gpt4o_base_tokenizer

        assert resolve_gpt4o_base_tokenizer() is mock_enc


def test_resolve_gpt4o_prefers_model_encoding() -> None:
    mock_enc = object()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("tiktoken.encoding_for_model", lambda *_: mock_enc)
        from tokenizer_utils import resolve_gpt4o_base_tokenizer

        assert resolve_gpt4o_base_tokenizer() is mock_enc


def test_extract_code_text_variants() -> None:
    ev = _load_eval_script()
    assert ev.extract_code_text({"content": " a "}) == " a "
    assert ev.extract_code_text({"code": "x"}) == "x"
    assert ev.extract_code_text({"text": "t"}) == "t"
    assert ev.extract_code_text({"other": 1}) is None


def test_collect_checkpoint_resume_mock_stream(tmp_path: Path) -> None:
    ev = _load_eval_script()
    corpus = tmp_path / "c.jsonl"
    ck = tmp_path / "ck.json"
    mf = tmp_path / "m.json"

    rows = [
        {"content": "def a():\n    return 1\n" * 5},
        {"content": "def b():\n    return 2\n" * 5},
    ]

    def fake_load(*_a, **_k):
        return iter(rows), "mock/dataset", []

    class _Tok:
        def encode(self, t):
            return list(range(len(t)))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "login_huggingface_if_needed", lambda: (False, []))
        mp.setattr(ev, "resolve_gpt4o_base_tokenizer", lambda: _Tok())
        mp.setattr(ev, "load_starcoder_python_stream", fake_load)
        mp.setattr(ev, "count_gpt4o_base_tokens", lambda text, encoder=None: max(1, len(text) // 10))
        s1, _m1 = ev.collect_starcoder_python_corpus(
            500,
            checkpoint_path=ck,
            manifest_path=mf,
            corpus_jsonl_path=corpus,
            resume=False,
            max_file_base_tokens=1_000_000,
        )
        assert len(s1) >= 1
        assert ck.exists()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "login_huggingface_if_needed", lambda: (False, []))
        mp.setattr(ev, "resolve_gpt4o_base_tokenizer", lambda: _Tok())
        mp.setattr(ev, "load_starcoder_python_stream", fake_load)
        mp.setattr(ev, "count_gpt4o_base_tokens", lambda text, encoder=None: max(1, len(text) // 10))
        s2, _m2 = ev.collect_starcoder_python_corpus(
            500,
            checkpoint_path=ck,
            manifest_path=mf,
            corpus_jsonl_path=corpus,
            resume=True,
            max_file_base_tokens=1_000_000,
        )
        assert len(s2) >= len(s1)


def test_run_stage1_only_no_stage23(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ev = _load_eval_script()
    calls: list[str] = []

    def no_stage2(*_a, **_k):
        calls.append("stage2")
        raise AssertionError("Stage2 must not run")

    def no_stage3(*_a, **_k):
        calls.append("stage3")
        raise AssertionError("Stage3 must not run")

    monkeypatch.setattr("pipeline.apply_stage2", no_stage2)
    monkeypatch.setattr("pipeline.apply_stage3", no_stage3)

    class Tok:
        def encode(self, text, allowed_special="all"):
            return text.split()

        @property
        def n_vocab(self):
            return 50000

    samples = ["def f():\n    return 1\n", "def g():\n    return 2\n"]
    r = ev.run_stage1_only_large_corpus(samples, tokenizer=Tok(), tok_type="tiktoken")
    assert "baseline_sequence_tokens" in r
    assert "stage1_sequence_tokens" in r
    assert "stage1_vocab_intro_tokens" in r
    assert "stage1_effective_total_tokens" in r
    assert calls == []


def test_write_stage1_large_corpus_reports_all_files(tmp_path: Path) -> None:
    ev = _load_eval_script()
    result = {
        "num_files": 1,
        "baseline_sequence_tokens": 10,
        "stage1_sequence_tokens": 8,
        "stage1_vocab_intro_tokens": 3,
        "stage1_effective_total_tokens": 11,
        "stage1_sequence_reduction_ratio": 0.2,
        "stage1_effective_reduction_ratio": -0.1,
        "selected_skeleton_count": 0,
        "selected_skeletons": [],
        "stage1_vocab_tokens": [],
        "skeleton_rows": [],
    }
    manifest = {"dataset_name": "test", "token_budget": 100, "actual_accumulated_base_tokens": 100}
    ev.write_stage1_large_corpus_reports(
        result,
        manifest=manifest,
        output_dir=tmp_path,
        tokenizer_name="test",
        resume_used=False,
        run_status="test",
    )
    assert (tmp_path / "stage1_starcoder_1m_summary.json").exists()
    assert (tmp_path / "stage1_starcoder_1m_summary.csv").exists()
    assert (tmp_path / "stage1_starcoder_1m_selected_skeletons.csv").exists()
    assert (tmp_path / "stage1_starcoder_1m_vocab_tokens.json").exists()
    assert (tmp_path / "stage1_starcoder_1m_manifest.json").exists()
