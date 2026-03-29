"""Shared types and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LiteralStat:
    """Per-literal statistics for one field."""

    literal: str
    count: int
    probability: float
    surprisal_bits: float
    raw_token_cost: int


@dataclass(slots=True)
class FieldProfile:
    """Distribution and cost profile of one field."""

    field_name: str
    sample_size: int
    cardinality: int
    entropy_bits: float
    expected_raw_token_cost: float
    stats: list[LiteralStat] = field(default_factory=list)


@dataclass(slots=True)
class CodeAssignment:
    """One literal-to-code assignment entry."""

    literal: str
    code: str
    raw_token_cost: int
    code_token_cost: int
    expected_gain: float


@dataclass(slots=True)
class FieldCodebook:
    """Field-level codebook."""

    field_name: str
    version: str
    assignments: list[CodeAssignment]
    escape_prefix: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FieldBuildResult:
    """Output bundle for one field build."""

    profile: FieldProfile
    codebook: FieldCodebook
    expected_coded_token_cost: float
    theoretical_headroom: float
    dictionary_coverage: float
    total_expected_gain: float
