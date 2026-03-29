"""Configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class SmoothingConfig:
    """Probability smoothing config."""

    method: str = "lidstone"
    alpha: float = 1.0


@dataclass(slots=True)
class CandidateSearchConfig:
    """Candidate short-code search config."""

    alphabet: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    max_code_length_chars: int = 4
    oversubscribe_factor: int = 8
    max_nodes_to_expand: int = 12000
    reserved_strings: tuple[str, ...] = ()


@dataclass(slots=True)
class AssignmentConfig:
    """Code assignment strategy config."""

    weight_mode: str = "p_times_cost"  # alternatives: "p_only"
    min_code_token_cost: int = 1
    min_gain: float = 0.0


@dataclass(slots=True)
class CompressionConfig:
    """Top-level system config."""

    random_seed: int = 7
    codebook_version: str = "v1"
    escape_prefix: str = "__L__"
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    candidate_search: CandidateSearchConfig = field(default_factory=CandidateSearchConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    fields: Sequence[str] = field(default_factory=tuple)
