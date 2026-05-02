"""Stage2 pre/post adaptation and lossy_cleaner safety helpers."""

from __future__ import annotations

import pytest

from lossy_cleaner import (
    CleaningConfig,
    clean_code,
    find_multiline_string_line_spans,
    is_preserved_directive_comment,
)
from pipeline import apply_stage1_stage2_adapted
from stage2.cleaning import run_stage2_post_surface, run_stage2_pre_safe
from stage2.config import build_stage2_execution_plan


def test_build_stage2_execution_plan_safe() -> None:
    p = build_stage2_execution_plan("safe")
    assert p.pre_cfg.remove_indentation is False
    assert p.pre_cfg.remove_docstrings is True
    assert p.post_cfg.remove_docstrings is False
    assert p.post_cfg.remove_comments is False
    assert p.post_cfg.remove_indentation is False


def test_build_stage2_execution_plan_aggressive() -> None:
    p = build_stage2_execution_plan("aggressive_upper_bound")
    assert p.pre_cfg.remove_indentation is False
    assert p.post_cfg.remove_indentation is True
    assert p.post_cfg.remove_docstrings is False


def test_run_stage2_pre_safe_docstring_and_comment() -> None:
    plan = build_stage2_execution_plan("safe")
    src = (
        'def _f():\n'
        '    """x"""\n'
        '    x = 1  # noqa: F401\n'
        '    return x\n'
    )
    out, stats = run_stage2_pre_safe(src, plan.pre_cfg, path="tests/t.py")
    assert '"""x"""' not in out
    assert "noqa" in out
    assert stats.removed_docstring_chars > 0


def test_run_stage2_post_surface_no_ast_on_syn() -> None:
    plan = build_stage2_execution_plan("safe")
    src = "<SYN_0>\ndef f():\n    return 1\n"
    out, stats = run_stage2_post_surface(src, plan.post_cfg, path="x.py")
    assert "<SYN_0>" in out
    assert stats.removed_docstring_chars == 0
    assert stats.docstring_removal_report is None


def test_run_stage2_post_surface_aggressive_strips_indent() -> None:
    plan = build_stage2_execution_plan("aggressive_upper_bound")
    src = "def f():\n    return 1\n"
    out, stats = run_stage2_post_surface(src, plan.post_cfg, path="x.py")
    assert stats.removed_indent_chars > 0
    assert not out.splitlines()[1].startswith("    ")


def test_apply_stage1_stage2_adapted_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def boom(*_a, **_k):
        calls.append("stage3")
        raise AssertionError("no stage3")

    monkeypatch.setattr("pipeline.apply_stage3", boom)

    class Repo:
        def skeleton_candidates(self):
            return []

    class Tok:
        def encode(self, text, allowed_special="all"):
            return list(text.encode("utf-8"))

        @property
        def n_vocab(self):
            return 256000

    src = "def f():\n    return 1\n"
    out = apply_stage1_stage2_adapted(
        src,
        Repo(),
        stage2_profile="safe",
        tokenizer=Tok(),
        tok_type="tiktoken",
        path="a.py",
    )
    assert calls == []
    assert out["stage2_order"] == "pre_safe -> stage1 -> post_surface"
    assert "stage2_pre_text" in out and "stage2_post_text" in out


def test_is_preserved_directive_comment() -> None:
    assert is_preserved_directive_comment(" noqa", 1, source_line="    # noqa")
    assert is_preserved_directive_comment(" type: ignore", 2, source_line=" # type: ignore")
    assert not is_preserved_directive_comment(" just a note", 3, source_line=" # note")


def test_multiline_string_lines_not_stripped() -> None:
    src = 'x = """a  \n\nb  \n"""\n'
    protected = find_multiline_string_line_spans(src)
    assert 1 in protected or 2 in protected
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=True,
        remove_trailing_whitespace=True,
        remove_docstrings=False,
        remove_indentation=False,
    )
    out, _ = clean_code(src, cfg)
    assert "a  " in out or "a" in out
