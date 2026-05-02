from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import analysis.run_alpha_rename_oracle_eval as run_alpha_rename_oracle_eval
import analysis.run_stage1_marker_ablation as run_stage1_marker_ablation
from marker_optimizer import build_marker_scheme
from offline_diagnostics import resolve_stage1_marker_scheme
from stage3.alpha_rename import alpha_rename_function_chunk


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
    ):
        assert field in summary


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
