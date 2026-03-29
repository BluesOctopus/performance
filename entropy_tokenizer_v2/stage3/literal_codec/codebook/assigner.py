"""Greedy prefix-free assigner."""

from __future__ import annotations

from dataclasses import dataclass

from literal_codec.codebook.candidate_pool import CandidatePoolGenerator
from literal_codec.codebook.optimizer import BaseCodeAssigner
from literal_codec.codebook.prefix_constraints import PrefixConstraintChecker
from literal_codec.config import AssignmentConfig, CandidateSearchConfig
from literal_codec.tokenizer.base import TokenizerAdapter
from literal_codec.types import CodeAssignment, FieldCodebook, FieldProfile


@dataclass(slots=True)
class GreedyPrefixFreeAssigner(BaseCodeAssigner):
    """
    Greedy assignment:
    1) collect low-cost prefix-feasible codes
    2) map high-weight literals to low-cost codes

    Why this mapping is reasonable:
    - Rearrangement / exchange argument: for w_a >= w_b and c_1 <= c_2,
      pairing (w_a,c_1),(w_b,c_2) is never worse than swapping to (w_a,c_2),(w_b,c_1).
    - Therefore sorting literals by descending weight and codes by ascending
      token cost gives an optimal pairing under a fixed feasible code set.
    """

    tokenizer: TokenizerAdapter
    candidate_config: CandidateSearchConfig
    assignment_config: AssignmentConfig
    escape_prefix: str
    version: str = "v1"

    def _literal_weight(self, probability: float, raw_token_cost: int) -> float:
        if self.assignment_config.weight_mode == "p_only":
            return probability
        return probability * raw_token_cost

    def build_codebook(self, profile: FieldProfile) -> tuple[FieldCodebook, float]:
        reserved = {item.literal for item in profile.stats}
        candidate_generator = CandidatePoolGenerator(
            tokenizer=self.tokenizer,
            config=self.candidate_config,
        )
        needed = len(profile.stats)
        candidates = candidate_generator.generate(
            needed=needed,
            escape_prefix=self.escape_prefix,
            reserved_strings=reserved,
        )
        checker = PrefixConstraintChecker(
            escape_prefix=self.escape_prefix,
            reserved_strings=reserved,
        )
        feasible_codes: list[tuple[str, int]] = []
        for cand in candidates:
            if checker.try_add(cand.code):
                feasible_codes.append((cand.code, cand.token_cost))
            if len(feasible_codes) >= needed:
                break

        literals = []
        for item in profile.stats:
            if item.raw_token_cost <= self.assignment_config.min_code_token_cost:
                continue
            weight = self._literal_weight(item.probability, item.raw_token_cost)
            literals.append((item.literal, item.probability, item.raw_token_cost, weight))

        literals.sort(key=lambda row: (-row[3], row[0]))
        feasible_codes.sort(key=lambda row: (row[1], len(row[0]), row[0]))

        assignments: list[CodeAssignment] = []
        literal_to_code_cost: dict[str, int] = {}
        for literal_row, code_row in zip(literals, feasible_codes, strict=False):
            literal, prob, raw_cost, _weight = literal_row
            code, code_cost = code_row
            gain = prob * max(0, raw_cost - code_cost)
            if gain <= self.assignment_config.min_gain:
                continue
            assignments.append(
                CodeAssignment(
                    literal=literal,
                    code=code,
                    raw_token_cost=raw_cost,
                    code_token_cost=code_cost,
                    expected_gain=gain,
                )
            )
            literal_to_code_cost[literal] = code_cost

        expected_coded_cost = 0.0
        for item in profile.stats:
            coded_cost = literal_to_code_cost.get(item.literal, item.raw_token_cost)
            expected_coded_cost += item.probability * coded_cost

        codebook = FieldCodebook(
            field_name=profile.field_name,
            version=self.version,
            assignments=assignments,
            escape_prefix=self.escape_prefix,
            metadata={
                "candidate_pool_size": len(candidates),
                "feasible_codes": len(feasible_codes),
                "assigned": len(assignments),
                "weight_mode": self.assignment_config.weight_mode,
            },
        )
        return codebook, expected_coded_cost
