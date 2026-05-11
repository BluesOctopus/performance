from __future__ import annotations

from types import SimpleNamespace

from config import (
    STAGE2_DEFAULT_MODE,
    STAGE2_DEFAULT_PROFILE,
    STAGE2_HYBRID_AB_MODE,
    STAGE2_HYBRID_AB_PROFILE,
)
from pipeline import resolve_stage2_for_pipeline


def test_hybrid_ab_implicit_stage2_uses_env_defaults() -> None:
    rc = SimpleNamespace(stage3_backend="hybrid_ab")
    p, m, src = resolve_stage2_for_pipeline(rc, None, None)
    assert src == "hybrid_ab_default"
    assert p == STAGE2_HYBRID_AB_PROFILE
    assert m == STAGE2_HYBRID_AB_MODE


def test_hybrid_ab_explicit_stage2_not_overridden() -> None:
    rc = SimpleNamespace(stage3_backend="hybrid_ab")
    p, m, src = resolve_stage2_for_pipeline(
        rc, STAGE2_DEFAULT_PROFILE, STAGE2_DEFAULT_MODE
    )
    assert src == "explicit"
    assert p == STAGE2_DEFAULT_PROFILE
    assert m == STAGE2_DEFAULT_MODE


def test_legacy_implicit_uses_global_defaults() -> None:
    rc = SimpleNamespace(stage3_backend="legacy")
    p, m, src = resolve_stage2_for_pipeline(rc, None, None)
    assert src == "global_default"
    assert p == STAGE2_DEFAULT_PROFILE
    assert m == STAGE2_DEFAULT_MODE
