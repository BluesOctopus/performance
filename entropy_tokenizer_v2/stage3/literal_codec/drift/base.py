"""Abstract interfaces for future B/C."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SemanticLossModel(ABC):
    """Future extension point for lossy compression (Plan B)."""

    @abstractmethod
    def loss(self, raw: str, compressed: str) -> float:
        """Return semantic loss score."""


class DriftDetector(ABC):
    """Future extension point for online drift detection (Plan C)."""

    @abstractmethod
    def should_retrain(self, old_distribution: Any, new_distribution: Any) -> bool:
        """Return True when retraining should be triggered."""


class CodebookSwitchPolicy(ABC):
    """Future extension point for version switching policy (Plan C)."""

    @abstractmethod
    def active_codebook(self, t: Any) -> str:
        """Return active codebook version id at time t."""
