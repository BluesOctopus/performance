"""Optimizer strategy extension points."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import FieldCodebook, FieldProfile


class BaseCodeAssigner(ABC):
    """Abstract strategy for assigning codes to literals."""

    @abstractmethod
    def build_codebook(self, profile: FieldProfile) -> tuple[FieldCodebook, float]:
        """Return (codebook, expected_coded_token_cost)."""


class BeamSearchAssigner(BaseCodeAssigner):
    """Placeholder for future beam-search assignment."""

    def build_codebook(self, profile: FieldProfile) -> tuple[FieldCodebook, float]:
        raise NotImplementedError("BeamSearchAssigner is reserved for future extension.")


class ExactAssigner(BaseCodeAssigner):
    """Placeholder for future exact (e.g., ILP/DP) assignment."""

    def build_codebook(self, profile: FieldProfile) -> tuple[FieldCodebook, float]:
        raise NotImplementedError("ExactAssigner is reserved for future extension.")
