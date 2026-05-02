from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import analysis.run_alpha_rename_oracle_eval as run_alpha_rename_oracle_eval
import analysis.run_offline_stage_ablation as run_offline_stage_ablation
import analysis.run_stage1_marker_ablation as run_stage1_marker_ablation
from marker_optimizer import build_marker_scheme
from offline_diagnostics import resolve_stage1_marker_scheme
from stage1_static.static_vocab import build_static_vocab_manifest
from stage3.alpha_pipeline import finalize_alpha_output
from stage3.alpha_rename import AlphaPassMetadata, AlphaPassResult, alpha_rename_function_chunk, apply_alpha_rename_pass
from training import build_stage2_alpha_data

_ADHERENCE_EVAL_SPEC = importlib.util.spec_from_file_location(
    "run_compression_adherence_eval",
    Path(__file__).resolve().parents[2] / "eval" / "run_compression_adherence_eval.py",
)
assert _ADHERENCE_EVAL_SPEC and _ADHERENCE_EVAL_SPEC.loader
run_compression_adherence_eval = importlib.util.module_from_spec(_ADHERENCE_EVAL_SPEC)
_ADHERENCE_EVAL_SPEC.loader.exec_module(run_compression_adherence_eval)


class PrefersSingleTokenEncoder:
    def encode(self, text: str, *args, **kwargs):
        if text.startswith("<SYN_"):
            return list(range(4))
        if text and text[0] in {"§", "¤", "@", "#", "$", "%", "&", "!"} and text[1:].isdigit():
            return [1]
        return text.split()


class CollisionEncoder(PrefersSingleTokenEncoder):
    pass


def test_marker_optimizer_prefers_single_token_marker() -> None:
    scheme = build_marker_scheme(
        "fake_tok",
        PrefersSingleTokenEncoder(),
        "tiktoken",
        "syn",
        4,
        avoid_texts=[],
    )
    assert scheme.namespace != "legacy"
    assert all(cost == 1 for cost in scheme.marker_token_costs)


def test_marker_optimizer_avoids_collisions() -> None:
    scheme = build_marker_scheme(
        "fake_tok",
        CollisionEncoder(),
        "tiktoken",
        "syn",
        2,
        avoid_texts=["contains §0 already"],
    )
    assert "§0" not in scheme.markers


def test_stage1_tokenizer_opt_marker_is_not_legacy_when_cheaper() -> None:
    scheme = resolve_stage1_marker_scheme(
        "tokenizer_opt",
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
        count=3,
        avoid_texts=[],
    )
    assert scheme.namespace != "legacy"
    assert all(not marker.startswith("<SYN_") for marker in scheme.markers)


def test_stage1_marker_ablation_outputs_marker_cost_fields(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "ablation"
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=PrefersSingleTokenEncoder(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f(x):\n    if x:\n        return x\n    return x\n",
            }
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage1_marker_ablation.py",
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
    run_stage1_marker_ablation.main()
    summary = (out_dir / "stage1_marker_ablation_summary.csv").read_text(encoding="utf-8")
    for field in (
        "stage1_marker_scheme",
        "marker_tokens_total",
        "avg_marker_token_cost",
        "legacy_marker_tokens_total",
        "tokenizer_opt_marker_tokens_total",
        "marker_saved_vs_legacy",
        "stage1_candidate_count",
        "stage1_selected_count",
        "stage1_selected_skeleton_count",
        "stage1_rejected_no_gain_count",
        "stage1_rejected_intro_cost_count",
        "stage1_rejected_marker_cost_count",
        "stage1_rejected_low_frequency_count",
        "stage1_apply_hit_count",
        "stage1_apply_miss_count",
        "per_chunk_effective_saved",
        "global_once_effective_saved",
        "pretrained_static_vocab_effective_saved",
        "break_even_codebook_tokens",
        "codebook_overhead_ratio",
        "stage1_ablation_invalid",
        "warning_type",
    ):
        assert field in summary


def test_stage1_2_equals_stage2_only_when_no_stage1_selected(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "ablation_noop"
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=PrefersSingleTokenEncoder(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f(arg_value):\n    return arg_value\n",
            }
        ],
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "build_stage_repo_config",
        lambda *args, **kwargs: SimpleNamespace(
            stage1_marker_scheme=kwargs.get("stage1_marker_scheme", "legacy"),
            stage1_candidate_stats=[],
            stage1_selected_stats=[],
            stage1_marker_tokens=[],
            stage1_rejected_no_gain_count=0,
            stage1_rejected_intro_cost_count=0,
            stage1_rejected_marker_cost_count=0,
            stage1_low_frequency_count=0,
            skeleton_candidates=lambda: [],
        ),
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "apply_stage2_pipeline",
        lambda text, **kwargs: {"stage2_pre_text": text, "stage2_post_text": "normalized"},
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "apply_stage1_stage2",
        lambda *args, **kwargs: {"stage1_text": args[0], "stage2_pre_text": args[0], "stage2_post_text": "normalized"},
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "apply_stage1_only",
        lambda text, *args, **kwargs: (text, {}),
    )
    monkeypatch.setattr(run_stage1_marker_ablation, "stage1_vocab_entries_for_text", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage1_marker_ablation.py",
            "--chunks",
            "dummy.jsonl",
            "--out_dir",
            str(out_dir),
        ],
    )
    run_stage1_marker_ablation.main()
    with (out_dir / "stage1_marker_ablation_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stage1_legacy_2 = next(row for row in rows if row["variant"] == "stage1_legacy_2")
    assert stage1_legacy_2["stage1_ablation_invalid"] == "False"
    assert stage1_legacy_2["warning_type"] == ""


def test_stage1_2_uses_identical_stage2_text_when_stage1_noop(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "ablation_hash"
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=PrefersSingleTokenEncoder(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f(arg_value):\n    return arg_value\n",
            }
        ],
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "build_stage_repo_config",
        lambda *args, **kwargs: SimpleNamespace(
            stage1_marker_scheme=kwargs.get("stage1_marker_scheme", "legacy"),
            stage1_candidate_stats=[],
            stage1_selected_stats=[],
            stage1_marker_tokens=[],
            stage1_rejected_no_gain_count=0,
            stage1_rejected_intro_cost_count=0,
            stage1_rejected_marker_cost_count=0,
            stage1_low_frequency_count=0,
            skeleton_candidates=lambda: [],
        ),
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "apply_stage2_pipeline",
        lambda text, **kwargs: {"stage2_pre_text": text, "stage2_post_text": "same_stage2"},
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "apply_stage1_stage2",
        lambda *args, **kwargs: {"stage1_text": args[0], "stage2_pre_text": args[0], "stage2_post_text": "same_stage2"},
    )
    monkeypatch.setattr(
        run_stage1_marker_ablation,
        "apply_stage1_only",
        lambda text, *args, **kwargs: (text, {}),
    )
    monkeypatch.setattr(run_stage1_marker_ablation, "stage1_vocab_entries_for_text", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage1_marker_ablation.py",
            "--chunks",
            "dummy.jsonl",
            "--out_dir",
            str(out_dir),
        ],
    )
    run_stage1_marker_ablation.main()
    with (out_dir / "stage1_marker_ablation_detail.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stage1_legacy_2 = next(row for row in rows if row["variant"] == "stage1_legacy_2")
    assert stage1_legacy_2["stage2_path_equal"] == "True"
    assert stage1_legacy_2["stage1_noop_but_stage2_diff_warning"] == "False"


def test_alpha_rename_summary_outputs_eligible_fields(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "alpha"
    monkeypatch.setattr(
        run_alpha_rename_oracle_eval,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=PrefersSingleTokenEncoder(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        run_alpha_rename_oracle_eval,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f(arg_name):\n    long_local_name = arg_name + 1\n    return long_local_name\n",
            }
        ],
    )
    monkeypatch.setattr(run_alpha_rename_oracle_eval, "apply_stage2_only", lambda text, **kwargs: text)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_alpha_rename_oracle_eval.py",
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
    run_alpha_rename_oracle_eval.main()
    summary = (out_dir / "alpha_rename_oracle_summary.csv").read_text(encoding="utf-8")
    for field in (
        "all_chunk_count",
        "eligible_function_count",
        "skipped_not_function_count",
        "skipped_parse_failed_count",
        "skipped_no_eligible_locals_count",
        "overall_hit_rate",
        "eligible_hit_rate",
        "overall_effective_saved",
        "eligible_effective_saved",
        "eligible_positive_doc_rate",
        "eligible_mean_delta",
        "renamed_count_total",
        "ast_equivalence_rate",
    ):
        assert field in summary


def test_alpha_rename_does_not_rename_args() -> None:
    result = alpha_rename_function_chunk(
        "def f(long_arg_name):\n    local_value = long_arg_name + 1\n    return local_value\n",
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
    )
    assert "long_arg_name" in result.renamed_text


def test_alpha_rename_does_not_rename_attributes() -> None:
    result = alpha_rename_function_chunk(
        "def f(obj):\n    temporary_value = obj.long_attribute_name\n    return temporary_value\n",
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
    )
    assert "long_attribute_name" in result.renamed_text


def test_alpha_rename_preserves_ast_parseability() -> None:
    result = alpha_rename_function_chunk(
        "def f(x):\n    local_result = x + 1\n    return local_result\n",
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
    )
    assert result.skipped_reason in {"", "no_eligible_locals"}
    assert result.ast_equivalent is True


def test_alpha_rename_can_reduce_long_local_names() -> None:
    result = alpha_rename_function_chunk(
        "def f(x):\n    intermediate_long_local_name = x + 1\n    return intermediate_long_local_name\n",
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
    )
    assert result.renamed_count >= 1
    assert result.delta_tokens >= 0


def test_alpha_rename_does_not_change_function_signature() -> None:
    result = apply_alpha_rename_pass(
        "def f(long_arg_name: int = 1) -> int:\n    intermediate_long_name = long_arg_name + 1\n    return intermediate_long_name\n",
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
    )
    assert result.metadata.alpha_public_signature_preserved is True


def test_alpha_rename_rolls_back_when_no_token_gain() -> None:
    class NoGainEncoder(PrefersSingleTokenEncoder):
        def encode(self, text: str, *args, **kwargs):
            return [0]

    result = apply_alpha_rename_pass(
        "def f(x):\n    long_name = x + 1\n    return long_name\n",
        tokenizer_name="fake_tok",
        encoder=NoGainEncoder(),
        tok_type="tiktoken",
    )
    assert result.metadata.alpha_applied is False
    assert result.metadata.alpha_rollback_reason == "no_token_gain"


def test_alpha_rename_compile_guardrail(monkeypatch) -> None:
    monkeypatch.setattr("stage3.alpha_rename._compile_source", lambda *args, **kwargs: (_ for _ in ()).throw(SyntaxError("boom")))
    result = apply_alpha_rename_pass(
        "def f(x):\n    very_long_local_name = x + 1\n    return very_long_local_name\n",
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
    )
    assert result.metadata.alpha_applied is False
    assert result.metadata.alpha_rollback_reason == "compile_failed_after_rename"


def test_variant_train_allowed_flags() -> None:
    assert run_offline_stage_ablation.variant_training_policy("stage2_alpha") == ("candidate", True, "")
    assert run_offline_stage_ablation.variant_training_policy("stage2_alpha_stage1_static") == ("candidate", True, "")
    assert run_offline_stage_ablation.variant_training_policy("stage2_alpha_stage1_tokenizer_opt")[1] is False
    assert run_offline_stage_ablation.variant_training_policy("stage2_3")[1] is False


def test_static_vocab_topk_has_positive_gain() -> None:
    manifest = build_static_vocab_manifest(
        [
            {"chunk_text": "def f(x):\n    if x:\n        return x\n    return x\n"},
            {"chunk_text": "def g(y):\n    if y:\n        return y\n    return y\n"},
            {"chunk_text": "def h(z):\n    if z:\n        return z\n    return z\n"},
        ],
        tokenizer_name="fake_tok",
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
        top_k=8,
    )
    assert all(float(entry["avg_token_gain"]) > 0 for entry in manifest["entries"])


def test_training_manifest_schema(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(
        build_stage2_alpha_data,
        "resolve_encoder_for_name",
        lambda _name: SimpleNamespace(tokenizer_name="fake_tok", encoder=PrefersSingleTokenEncoder(), tok_type="tiktoken"),
    )
    monkeypatch.setattr(
        build_stage2_alpha_data,
        "load_chunks",
        lambda _path: [
            {
                "chunk_id": "c1",
                "source_id": "s.py",
                "symbol_type": "function",
                "symbol_name": "f",
                "chunk_text": "def f(x):\n    long_name = x + 1\n    return long_name\n",
            }
        ],
    )
    monkeypatch.setattr(build_stage2_alpha_data, "apply_stage2_only", lambda text, **kwargs: text)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_stage2_alpha_data.py",
            "--chunks",
            "dummy.jsonl",
            "--output",
            str(out_path),
            "--filter-mode",
            "all_safe",
        ],
    )
    build_stage2_alpha_data.main()
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(row.keys()) == {
        "raw_text",
        "compressed_text",
        "variant",
        "tokenizer_name",
        "raw_tokens",
        "compressed_tokens",
        "effective_saved",
        "alpha_metadata",
        "static_vocab_metadata",
        "safety_checks",
        "split",
    }


def test_finalize_alpha_output_rolls_back_to_safe_stage2_text() -> None:
    alpha_result = AlphaPassResult(
        output_text="def broken(:\n    pass\n",
        metadata=AlphaPassMetadata(
            alpha_applied=True,
            alpha_renamed_count=1,
            alpha_raw_tokens=10,
            alpha_tokens=8,
            alpha_delta_tokens=2,
            alpha_guardrail_triggered=False,
            alpha_rollback_reason="",
            alpha_skipped_reason="",
            alpha_ast_ok=False,
            alpha_compile_ok=False,
            alpha_public_signature_preserved=False,
        ),
    )
    finalized = finalize_alpha_output(
        "def f(x):\n    return x\n",
        alpha_result,
        encoder=PrefersSingleTokenEncoder(),
        tok_type="tiktoken",
    )
    assert finalized.compressed_text == "def f(x):\n    return x\n"
    assert finalized.alpha_metadata["alpha_applied"] is False
    assert finalized.alpha_metadata["alpha_guardrail_triggered"] is True
    assert finalized.safety_checks["compressed_parse_ok"] is True
    assert finalized.safety_checks["compressed_compile_ok"] is True
    assert finalized.safety_checks["alpha_public_signature_preserved"] is True


def test_adherence_eval_uses_decoded_checks_for_static_variant() -> None:
    summary = run_compression_adherence_eval.build_summary(
        [
            {
                "variant": "stage2_alpha_stage1_static",
                "compressed_text": "<SYN_0> payload",
                "effective_saved": 5,
                "raw_tokens": 10,
                "compressed_tokens": 6,
                "alpha_metadata": {
                    "alpha_public_signature_preserved": True,
                    "alpha_guardrail_triggered": False,
                    "alpha_rollback_reason": "",
                    "alpha_skipped_reason": "",
                    "alpha_ast_ok": True,
                    "alpha_compile_ok": True,
                },
                "static_vocab_metadata": {
                    "encoded_parse_ok": False,
                    "decode_success": True,
                    "decoded_parse_ok": True,
                    "decoded_compile_ok": True,
                    "decoded_ast_equivalent": True,
                },
            }
        ]
    )
    assert summary["invalid_python_rate"] == 0.0
    assert summary["decoded_invalid_python_rate"] == 0.0
    assert summary["static_vocab_decode_success"] == 1.0
