from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import analysis.run_offline_stage_ablation as run_ablation
import analysis.stage1_roundtrip_eval as stage1_roundtrip_eval
import analysis.stage3_gain_eval as stage3_gain_eval
from stage3.guardrail import apply_effective_guardrail


def test_guardrail_rolls_back_when_candidate_not_smaller() -> None:
    decision = apply_effective_guardrail(
        baseline_effective_tokens=10,
        candidate_effective_tokens=10,
        rollback_label="rollback_to_stage2",
    )
    assert decision.should_rollback is True
    assert decision.reason == "rollback_to_stage2: candidate_effective_tokens >= baseline_effective_tokens"


def test_stage3_gain_eval_does_not_fake_decode_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_path = tmp_path / "stage3.csv"
    monkeypatch.setattr(
        stage3_gain_eval,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=object(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        stage3_gain_eval,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f():\n    return 1\n",
            }
        ],
    )
    monkeypatch.setattr(stage3_gain_eval, "build_stage_repo_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(stage3_gain_eval, "apply_stage2_only", lambda text, **kwargs: text)
    monkeypatch.setattr(
        stage3_gain_eval,
        "apply_stage3",
        lambda *args, **kwargs: {
            "text": "<A>",
            "vocab_entries": [{"token": "<A>", "definition": "x"}],
            "stage3_triggered": True,
            "stage3_candidate_count": 1,
            "stage3_selected_count": 1,
            "projected_gain": 2,
        },
    )
    monkeypatch.setattr(
        stage3_gain_eval,
        "build_token_ledger",
        lambda raw_text, compressed_text, **kwargs: SimpleNamespace(
            tokenizer_name=kwargs["tokenizer_name"],
            raw_tokens=10,
            compressed_tokens=8 if compressed_text != "<A>" else 9,
            codebook_tokens=1 if compressed_text == "<A>" else 0,
            wrapper_tokens=0,
            task_prompt_tokens=0,
            effective_tokens=8 if compressed_text != "<A>" else 10,
            effective_prompt_tokens=8 if compressed_text != "<A>" else 10,
            gross_saved=2 if compressed_text != "<A>" else 1,
            net_saved=2 if compressed_text != "<A>" else 0,
            effective_saved=2 if compressed_text != "<A>" else 0,
        ),
    )
    monkeypatch.setattr(
        stage3_gain_eval,
        "with_guardrail",
        lambda **kwargs: (
            SimpleNamespace(
                tokenizer_name=kwargs["tokenizer_name"],
                raw_tokens=10,
                compressed_tokens=8,
                codebook_tokens=0,
                wrapper_tokens=0,
                task_prompt_tokens=0,
                effective_tokens=8,
                effective_prompt_tokens=8,
                gross_saved=2,
                net_saved=2,
                effective_saved=2,
            ),
            SimpleNamespace(should_rollback=True, reason="rollback_to_stage2"),
        ),
    )
    monkeypatch.setattr(
        stage3_gain_eval,
        "stage3_decode_status",
        lambda **kwargs: {
            "decode_success": False,
            "roundtrip_ok": False,
            "ast_equivalent": False,
            "error_type": "stage3_decoder_missing",
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "stage3_gain_eval.py",
            "--chunks",
            "dummy.jsonl",
            "--out",
            str(out_path),
            "--tokenizer",
            "gpt4",
            "--stage2-profile",
            "aggressive",
        ],
    )
    stage3_gain_eval.main()
    text = out_path.read_text(encoding="utf-8")
    assert "stage3_decoder_missing" in text
    assert ",False,False,False," in text


def test_run_offline_stage_ablation_outputs_required_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_dir = tmp_path / "ablation"
    monkeypatch.setattr(
        run_ablation,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=object(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        run_ablation,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f():\n    return 1\n",
            }
        ],
    )
    monkeypatch.setattr(run_ablation, "build_stage_repo_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run_ablation,
        "run_variant",
        lambda variant, chunk, repo_config, tokenizer_name, encoder, tok_type, stage2_profile, codebook_accounting_mode: {
            "variant": variant,
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "symbol_type": chunk["symbol_type"],
            "symbol_name": chunk["symbol_name"],
            "tokenizer_name": tokenizer_name,
            "stage2_profile": stage2_profile,
            "codebook_accounting_mode": codebook_accounting_mode,
            "raw_tokens": 10,
            "compressed_tokens": 8,
            "codebook_tokens": 1,
            "wrapper_tokens": 0,
            "task_prompt_tokens": 0,
            "effective_tokens": 9,
            "effective_prompt_tokens": 9,
            "gross_saved": 2,
            "net_saved": 1,
            "effective_saved": 1,
            "stage1_hit": variant in {"stage1_only", "stage1_2", "stage1_2_3"},
            "stage3_triggered": variant in {"stage3_only", "stage2_3", "stage1_2_3"},
            "roundtrip_ok": False,
            "ast_equivalent": False,
            "decode_success": False,
            "error_type": "",
            "rollback_applied": False,
            "rollback_reason": "",
            "_codebook_entries": [{"token": "<A>", "definition": "x"}],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_offline_stage_ablation.py",
            "--chunks",
            "dummy.jsonl",
            "--out_dir",
            str(out_dir),
            "--tokenizer",
            "gpt4",
            "--stage2-profile",
            "aggressive",
        ],
    )
    run_ablation.main()
    summary = (out_dir / "stage_ablation_offline_summary.csv").read_text(encoding="utf-8")
    for field in (
        "tokenizer_name",
        "stage2_profile",
        "codebook_accounting_mode",
        "raw_tokens",
        "compressed_tokens",
        "codebook_tokens",
        "wrapper_tokens",
        "effective_tokens",
        "gross_saved",
        "net_saved",
        "effective_saved",
        "stage1_hit_rate",
        "stage3_trigger_rate",
        "roundtrip_success_rate",
        "ast_equivalence_rate",
        "decode_success_rate",
    ):
        assert field in summary


def test_tokenizer_name_is_recorded(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "stage1.csv"
    monkeypatch.setattr(
        stage1_roundtrip_eval,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=object(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        stage1_roundtrip_eval,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f():\n    return 1\n",
            }
        ],
    )
    monkeypatch.setattr(stage1_roundtrip_eval, "build_stage_repo_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(stage1_roundtrip_eval, "apply_stage1_with_stats", lambda *args, **kwargs: ("<SYN_0> x", {}))
    monkeypatch.setattr(stage1_roundtrip_eval, "stage1_vocab_entries_for_text", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        stage1_roundtrip_eval,
        "decode_stage1_text",
        lambda *args, **kwargs: SimpleNamespace(
            roundtrip_ok=False,
            decode_success=False,
            ast_equivalent=False,
            error_type="ambiguous_slot_boundaries",
        ),
    )
    monkeypatch.setattr(
        stage1_roundtrip_eval,
        "build_token_ledger",
        lambda *args, **kwargs: SimpleNamespace(
            tokenizer_name=kwargs["tokenizer_name"],
            raw_tokens=10,
            compressed_tokens=8,
            codebook_tokens=0,
            wrapper_tokens=0,
            task_prompt_tokens=0,
            effective_tokens=8,
            effective_prompt_tokens=8,
            gross_saved=2,
            net_saved=2,
            effective_saved=2,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "stage1_roundtrip_eval.py",
            "--chunks",
            "dummy.jsonl",
            "--out",
            str(out_path),
            "--tokenizer",
            "gpt4",
        ],
    )
    stage1_roundtrip_eval.main()
    text = out_path.read_text(encoding="utf-8")
    assert "tokenizer_name" in text
    assert "fake_tok" in text
