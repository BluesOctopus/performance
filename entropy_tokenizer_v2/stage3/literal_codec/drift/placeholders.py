"""Concrete placeholders for interfaces."""

from __future__ import annotations

from typing import Any

from .base import CodebookSwitchPolicy, DriftDetector, SemanticLossModel


class ZeroSemanticLossModel(SemanticLossModel):
    """Plan A uses strict reversibility, so semantic loss is always zero."""

    def loss(self, raw: str, compressed: str) -> float:
        return 0.0


class NoopDriftDetector(DriftDetector):
    """Placeholder detector that never retrains."""

    def should_retrain(self, old_distribution: Any, new_distribution: Any) -> bool:
        return False


class StaticCodebookSwitchPolicy(CodebookSwitchPolicy):
    """Placeholder policy that always returns one version."""

    def __init__(self, version: str) -> None:
        self._version = version

    def active_codebook(self, t: Any) -> str:
        return self._version
