"""Conservative Stage2 docstring removal (AST-only, safe_only)."""

from __future__ import annotations

from lossy_cleaner import CleaningConfig, clean_code
from stage2.docstring_analysis import (
    classify_docstring_path_context,
    has_structured_doc_markers,
    iter_docstrings,
    normalize_path_for_docstring_policy,
    remove_safe_docstrings,
)


def test_module_docstring_preserved_private_helper_may_be_removed() -> None:
    src = '''"""module docs"""

def _helper():
    """internal"""
    return 1
'''
    out, rep = remove_safe_docstrings(src)
    assert '"""module docs"""' in out
    assert rep["parse_failed"] is False
    assert any(k["qualname"] == "<module>" for k in rep["kept"])
    removed_qn = {r["qualname"] for r in rep["removed"]}
    assert "_helper" in removed_qn


def test_public_function_docstring_preserved() -> None:
    src = '''def foo(x):
    """Return x."""
    return x
'''
    out, rep = remove_safe_docstrings(src)
    assert '"""Return x."""' in out
    assert rep["removed_count"] == 0
    assert any("foo" in k["qualname"] for k in rep["kept"])


def test_private_function_docstring_removed_when_safe() -> None:
    src = '''def _helper(x):
    """Internal helper."""
    return x
'''
    out, rep = remove_safe_docstrings(src)
    assert '"""Internal helper."""' not in out
    assert rep["removed_count"] >= 1


def test_structured_private_docstring_preserved() -> None:
    src = '''def _helper(x):
    """
    Args:
        x: value
    Returns:
        value
    """
    return x
'''
    out, rep = remove_safe_docstrings(src)
    assert "Args:" in out
    assert rep["removed_count"] == 0


def test_file_uses___doc___conservative_keep_all() -> None:
    src = '''def _a():
    """x"""
    return 1

y = foo.__doc__
'''
    out, rep = remove_safe_docstrings(src)
    assert '"""x"""' in out
    assert rep["removed_count"] == 0


def test_inspect_getdoc_signal_keeps_docstrings() -> None:
    src = '''import inspect

def _a():
    """keep me"""
    return 1

inspect.getdoc(_a)
'''
    out, rep = remove_safe_docstrings(src)
    assert "keep me" in out
    assert rep["removed_count"] == 0


def test_high_risk_decorator_preserves_docstring() -> None:
    src = '''@app.get("/x")
def foo():
    """endpoint docs"""
    return 1
'''
    out, rep = remove_safe_docstrings(src)
    assert "endpoint docs" in out
    assert rep["removed_count"] == 0


def test_parse_failed_no_regex_fallback() -> None:
    src = "def broken(\n"
    out, rep = remove_safe_docstrings(src)
    assert out == src
    assert rep["parse_failed"] is True
    assert rep["removed_count"] == 0


def test_triple_quoted_assignment_not_docstring() -> None:
    src = '''x = """
hello
world
"""
'''
    out, rep = remove_safe_docstrings(src)
    assert "hello" in out
    assert rep["removed_count"] == 0


def test_has_structured_doc_markers() -> None:
    assert has_structured_doc_markers("Args:\\n  x: int")
    assert has_structured_doc_markers(":param foo: bar")
    assert not has_structured_doc_markers("just some text")


def test_clean_code_uses_safe_docstring_path() -> None:
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=False,
        remove_trailing_whitespace=False,
        remove_docstrings=True,
        docstring_removal_mode="safe_only",
        remove_indentation=False,
    )
    src = "def _x():\n    '''d'''\n    return 1\n"
    out, stats = clean_code(src, cfg)
    assert stats.docstring_removal_report is not None
    assert stats.docstring_removal_report["parse_failed"] is False
    assert "'''d'''" not in out


def test_iter_docstrings_finds_module_and_function() -> None:
    src = '"""mod"""\ndef f():\n    """f"""\n    pass\n'
    ds = iter_docstrings(src)
    kinds = {d.kind for d in ds}
    assert "module" in kinds
    assert "function" in kinds


def test_normalize_path_for_docstring_policy() -> None:
    assert normalize_path_for_docstring_policy(None) == ""
    assert "tests" in normalize_path_for_docstring_policy(r"E:\Repo\Tests\X.py")


def test_nested_public_helper_docstring_removed_without_path() -> None:
    src = '''def outer():
    def helper():
        """local helper"""
        return 1
    return helper()
'''
    out, rep = remove_safe_docstrings(src)
    assert "local helper" not in out
    assert rep["removed_count"] >= 1
    assert any("nested_low_risk" in r for rm in rep["removed"] for r in rm.get("risk_reasons", []))


def test_nested_structured_docstring_preserved() -> None:
    src = '''def outer():
    def helper():
        """
        Args:
            x: value
        """
        return 1
    return helper()
'''
    out, rep = remove_safe_docstrings(src)
    assert "Args:" in out
    assert rep["removed_count"] == 0


def test_tests_path_private_helper_removed_and_path_in_report() -> None:
    src = '''def _helper():
    """internal helper"""
    return 1
'''
    out, rep = remove_safe_docstrings(src, path="tests/test_utils.py")
    assert "internal helper" not in out
    pc = rep["path_context"]
    assert pc["is_low_risk_context"] is True
    assert "tests" in pc["matched_context_labels"]
    assert "tests/test_utils.py" in pc["normalized_path"]
    assert any(
        any("low_risk_path_context:tests" in r for r in rm.get("risk_reasons", []))
        for rm in rep["removed"]
    )


def test_scripts_path_private_docstring_removed() -> None:
    src = '''def _build():
    """cache step"""
    return 0
'''
    out, rep = remove_safe_docstrings(src, path="scripts/build_cache.py")
    assert "cache step" not in out
    assert "scripts" in rep["path_context"]["matched_context_labels"]


def test_tests_path_public_nested_helper_docstring_preserved() -> None:
    src = '''def outer():
    def helper():
        """public helper docs"""
        return 1
    return helper()
'''
    out, rep = remove_safe_docstrings(src, path="tests/test_utils.py")
    assert "public helper docs" in out
    assert rep["removed_count"] == 0


def test_runtime_signal_keeps_docstrings_even_in_tests_path() -> None:
    src = '''def _helper():
    """docs"""
    return 1

x = _helper.__doc__
'''
    out, rep = remove_safe_docstrings(src, path="tests/test_utils.py")
    assert "docs" in out
    assert rep["removed_count"] == 0


def test_clean_code_passes_path_to_safe_removal() -> None:
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=False,
        remove_trailing_whitespace=False,
        remove_docstrings=True,
        docstring_removal_mode="safe_only",
        remove_indentation=False,
    )
    src = "def _x():\n    '''d'''\n    return 1\n"
    direct, dr = remove_safe_docstrings(src, path="tests/test_x.py")
    via, stats = clean_code(src, cfg, path="tests/test_x.py")
    assert stats.docstring_removal_report["path_context"] == dr["path_context"]
    assert "'''d'''" not in via
    assert "'''d'''" not in direct


def test_classify_docstring_path_context_labels() -> None:
    ctx = classify_docstring_path_context("proj/internal/pkg/mod.py")
    assert ctx["is_internal_file"] is True
    assert "internal" in ctx["matched_context_labels"]
