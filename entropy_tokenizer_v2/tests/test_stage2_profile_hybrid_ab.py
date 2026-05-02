from __future__ import annotations

import pytest

from stage2.config import build_stage2_config


def test_stage2_hybrid_ab_aggressive_builds_blockwise() -> None:
    cfg = build_stage2_config(profile="stage2_hybrid_ab_aggressive", mode="blockwise")
    assert cfg.profile == "stage2_hybrid_ab_aggressive"
    assert cfg.mode == "blockwise"
    assert cfg.cleaning.remove_comments is True
    assert cfg.cleaning.remove_docstrings is True
    assert cfg.cleaning.remove_indentation is True


def test_stage2_hybrid_ab_aggressive_rejects_linewise_docstrings() -> None:
    with pytest.raises(ValueError, match="linewise"):
        build_stage2_config(profile="stage2_hybrid_ab_aggressive", mode="linewise")
