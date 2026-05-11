"""Frequency and probability estimation."""

from __future__ import annotations

from collections import Counter


def count_literals(values: list[str]) -> Counter[str]:
    """Count literal frequencies."""
    return Counter(values)


def empirical_distribution(counts: Counter[str]) -> dict[str, float]:
    """Maximum-likelihood empirical probabilities p_hat(x)=n(x)/N."""
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def lidstone_distribution(counts: Counter[str], alpha: float = 1.0) -> dict[str, float]:
    """
    Lidstone-smoothed distribution.
    Laplace smoothing is alpha=1.
    """
    if not counts:
        return {}
    vocab_size = len(counts)
    total = sum(counts.values())
    denominator = total + alpha * vocab_size
    if denominator <= 0:
        return empirical_distribution(counts)
    return {k: (v + alpha) / denominator for k, v in counts.items()}
