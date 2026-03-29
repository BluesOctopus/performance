from __future__ import annotations

import sys
from pathlib import Path

_S3 = Path(__file__).resolve().parents[1]
if str(_S3) not in sys.path:
    sys.path.insert(0, str(_S3))

from router import ABRoutingConfig, classify_string_kind, route_name_literal


def test_router_name_goes_to_a():
    assert route_name_literal("variable") == "A"
    assert route_name_literal("attribute") == "A"


def test_router_exact_strings_go_to_a():
    cfg = ABRoutingConfig()
    assert classify_string_kind(r"'/tmp/data/file.json'", cfg) == "A"
    assert classify_string_kind(r"'[A-Za-z_]+'", cfg) == "A"
    assert classify_string_kind(r"'effective_total_reduction_pct'", cfg) == "A"


def test_router_free_text_goes_to_b():
    cfg = ABRoutingConfig(free_text_min_chars=20, free_text_min_words=4)
    s = "'This is a long natural language sentence for semantic grouping.'"
    assert classify_string_kind(s, cfg) == "B"


def test_router_unknown_fallback():
    cfg = ABRoutingConfig()
    assert classify_string_kind("'x'", cfg) == "fallback"
