from __future__ import annotations

from stage3.routing.router import ABRoutingConfig, classify_string_with_reason


def test_router_mid_free_text_lane_enabled() -> None:
    cfg = ABRoutingConfig(
        free_text_min_chars=24,
        free_text_min_words=4,
        enable_mid_free_text=True,
        free_text_mid_min_chars=12,
        free_text_mid_min_words=3,
    )
    route, reason = classify_string_with_reason("'retry request failed'", cfg)
    assert route == "B"
    assert reason == "mid_free_text"


def test_router_multiline_whitelist_lane() -> None:
    cfg = ABRoutingConfig(
        allow_multiline_whitelist=True,
        multiline_max_lines=3,
        multiline_max_chars=220,
        enable_mid_free_text=True,
    )
    route, reason = classify_string_with_reason("'line one\\nline two'", cfg)
    assert route == "B"
    assert reason == "multiline_whitelist"


def test_router_multiline_reject_when_disabled() -> None:
    cfg = ABRoutingConfig(allow_multiline_whitelist=False)
    route, reason = classify_string_with_reason("'line one\\nline two'", cfg)
    assert route == "fallback"
    assert reason == "multiline_disabled"
