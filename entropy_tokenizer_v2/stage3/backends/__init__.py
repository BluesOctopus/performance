from __future__ import annotations

from .base import Stage3Backend, Stage3EncodeResult
from .legacy_backend import LegacyStage3Backend
from .plan_a_backend import PlanAStage3Backend
from .hybrid_ab_backend import HybridABStage3Backend

_BACKENDS: dict[str, Stage3Backend] = {
    "legacy": LegacyStage3Backend(),
    "plan_a": PlanAStage3Backend(),
    "hybrid_ab": HybridABStage3Backend(),
}


def get_stage3_backend(name: str) -> Stage3Backend:
    key = (name or "").strip().lower()
    return _BACKENDS.get(key, _BACKENDS["legacy"])


__all__ = ["Stage3Backend", "Stage3EncodeResult", "get_stage3_backend"]

