"""Plan A string literal heuristics."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from literal_codec.pipeline.string_filter import (
    StringFilterConfig,
    string_literal_should_drop,
)


def test_strict_drops_doc_and_regex() -> None:
    cfg = StringFilterConfig(min_count=1, min_raw_token_cost=0, strict_heuristics=True)
    inner = "Stage 1 and Stage 2 pipeline effective_total reduction"
    drop, r = string_literal_should_drop(
        '"x"', inner=inner, count=5, raw_token_cost=50, cfg=cfg
    )
    assert drop and r == "doc_marker"

    inner2 = r"foo \b bar \d+ spam [^\\w\\s] extra tail " + "x" * 30
    drop2, r2 = string_literal_should_drop(
        '"x"', inner=inner2, count=3, raw_token_cost=40, cfg=cfg
    )
    assert drop2 and r2 == "regex_like"


def test_keylike_short_string_kept() -> None:
    cfg = StringFilterConfig(min_count=2, min_raw_token_cost=4, strict_heuristics=True)
    inner = "api_v1"
    drop, _ = string_literal_should_drop(
        '"api_v1"', inner=inner, count=10, raw_token_cost=12, cfg=cfg
    )
    assert not drop
