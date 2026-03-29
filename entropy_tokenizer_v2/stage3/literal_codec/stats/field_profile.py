"""Field profile builder."""

from __future__ import annotations

from dataclasses import dataclass

from literal_codec.config import SmoothingConfig
from literal_codec.stats.entropy import entropy_bits, surprisal_bits
from literal_codec.stats.frequency import count_literals, empirical_distribution, lidstone_distribution
from literal_codec.tokenizer.base import TokenizerAdapter
from literal_codec.types import FieldProfile, LiteralStat


@dataclass(slots=True)
class FieldProfiler:
    """Compute frequency/distribution/entropy/cost profile for a field."""

    tokenizer: TokenizerAdapter
    smoothing: SmoothingConfig

    def _distribution(self, values: list[str]) -> tuple[dict[str, int], dict[str, float]]:
        counts = count_literals(values)
        if self.smoothing.method in {"laplace", "lidstone"}:
            probs = lidstone_distribution(counts, alpha=self.smoothing.alpha)
        else:
            probs = empirical_distribution(counts)
        return dict(counts), probs

    def build(self, field_name: str, values: list[str]) -> FieldProfile:
        sample_size = len(values)
        if sample_size == 0:
            return FieldProfile(
                field_name=field_name,
                sample_size=0,
                cardinality=0,
                entropy_bits=0.0,
                expected_raw_token_cost=0.0,
                stats=[],
            )

        counts, probs = self._distribution(values)
        h = entropy_bits(probs)
        stats: list[LiteralStat] = []
        expected_cost = 0.0
        for literal in sorted(counts):
            p = probs[literal]
            raw_cost = self.tokenizer.token_length(literal)
            expected_cost += p * raw_cost
            stats.append(
                LiteralStat(
                    literal=literal,
                    count=counts[literal],
                    probability=p,
                    surprisal_bits=surprisal_bits(p),
                    raw_token_cost=raw_cost,
                )
            )

        stats.sort(key=lambda x: (-x.count, x.literal))
        return FieldProfile(
            field_name=field_name,
            sample_size=sample_size,
            cardinality=len(counts),
            entropy_bits=h,
            expected_raw_token_cost=expected_cost,
            stats=stats,
        )
