"""Entropy and surprisal utilities."""

from __future__ import annotations

import math


def surprisal_bits(probability: float) -> float:
    """I(x)=-log2 p(x)."""
    if probability <= 0:
        return float("inf")
    return -math.log2(probability)


def entropy_bits(distribution: dict[str, float]) -> float:
    """
    Shannon entropy: H(X) = -sum p(x) log2 p(x).
    This is a bit-level lower-bound reference for optimal prefix coding.
    """
    h = 0.0
    for p in distribution.values():
        if p > 0:
            h -= p * math.log2(p)
    return h
