"""Post-prune entry point smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from literal_codec.pipeline.plan_a_post_prune import prune_plan_a_field_results


def test_prune_disabled_returns_input():
    books, frs, rep = prune_plan_a_field_results(
        [],
        {},
        tokenizer=None,
        tok_type="",
        escape_prefix="__L__",
        enabled=False,
    )
    assert rep["stage3_plan_a_post_prune_enabled"] is False
    assert books == {}
    assert frs == []
