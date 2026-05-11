import pytest

from lossy_cleaner import CleaningConfig
from stage2.cleaning import stage2_clean_skip_syn, stage2_clean_skip_syn_and_stats
from stage2.config import build_stage2_config


def test_build_stage2_config_rejects_docstrings_in_linewise() -> None:
    with pytest.raises(ValueError, match="linewise mode does not support remove_docstrings=True"):
        build_stage2_config(
            profile="stage2_aggressive",
            mode="linewise",
            overrides={"remove_docstrings": True},
        )


def test_stage2_clean_rejects_docstrings_in_linewise() -> None:
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=True,
        remove_trailing_whitespace=True,
        remove_docstrings=True,
        remove_indentation=False,
    )
    with pytest.raises(ValueError, match="linewise mode does not support remove_docstrings=True"):
        stage2_clean_skip_syn('"""doc"""', cfg, mode="linewise")


def test_stage2_clean_skip_syn_and_stats_docstring_report_has_path_context() -> None:
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=True,
        remove_trailing_whitespace=True,
        remove_docstrings=True,
        remove_indentation=False,
    )
    src = '''def _h():
    """x"""
    return 1
'''
    _, stats = stage2_clean_skip_syn_and_stats(
        src,
        cfg,
        mode="blockwise",
        path="tests/test_utils.py",
    )
    report = stats.docstring_removal_report
    assert report is not None
    pc = report["path_context"]
    assert pc["normalized_path"] == "tests/test_utils.py"
    assert "tests" in pc["matched_context_labels"]
    assert pc["is_low_risk_context"] is True
