from __future__ import annotations

from stage3.router import (
    ABRoutingConfig,
    classify_string_kind,
    classify_string_with_reason,
    route_name_literal,
)


def test_router_name_goes_to_a():
    assert route_name_literal("variable") == "A"
    assert route_name_literal("attribute") == "A"


def test_router_exact_strings_go_to_a():
    cfg = ABRoutingConfig()
    assert classify_string_kind(r"'/tmp/data/file.json'", cfg) == "A"
    assert classify_string_kind(r"'[A-Za-z_]+'", cfg) == "A"
    assert classify_string_kind(r"'db_config_key'", cfg) == "A"


def test_router_free_text_goes_to_b():
    cfg = ABRoutingConfig(free_text_min_chars=20, free_text_min_words=4)
    s = "'This is a long natural language sentence for semantic grouping.'"
    assert classify_string_kind(s, cfg) == "B"


def test_router_unknown_fallback():
    cfg = ABRoutingConfig()
    assert classify_string_kind("'x'", cfg) == "fallback"
    route, reason = classify_string_with_reason("'x'", cfg)
    assert route == "fallback"
    assert reason == "short_literal"


def test_router_key_like_from_config_and_reason():
    cfg = ABRoutingConfig(key_like_patterns=(r"^my-custom-key$",))
    route, reason = classify_string_with_reason("'my-custom-key'", cfg)
    assert route == "A"
    assert reason == "key_like"
