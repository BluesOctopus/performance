"""Greedy prefix-free assigner (real surface-form token costs)."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_pool import CandidatePoolGenerator
from .optimizer import BaseCodeAssigner
from .prefix_constraints import PrefixConstraintChecker
from ..config import AssignmentConfig, CandidateSearchConfig
from ..pipeline.surface_cost import encoded_form_token_cost
from ..tokenizer.base import TokenizerAdapter
from ..types import CodeAssignment, FieldCodebook, FieldProfile


@dataclass(slots=True)
class GreedyPrefixFreeAssigner(BaseCodeAssigner):
    """
    Greedy assignment with **encoded surface** token costs (not bare code length):

    - NAME fields: ``{escape}{V|A}{code}``
    - STRING field: ``repr(f"{escape}S{code}")`` (must match ``source_codec``)

    Prefix-free constraints apply to bare *code* strings only.
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
        field_name = profile.field_name
        max_assign = self.assignment_config.max_assignments_by_field.get(field_name, 10**9)

        reserved = {item.literal for item in profile.stats}
        candidate_generator = CandidatePoolGenerator(
            tokenizer=self.tokenizer,
            config=self.candidate_config,
        )
        pool_needed = len(profile.stats)
        candidates = candidate_generator.generate(
            needed=pool_needed,
            escape_prefix=self.escape_prefix,
            reserved_strings=reserved,
        )

        scored: list[tuple[str, int]] = []
        for cand in candidates:
            enc_cost = encoded_form_token_cost(
                field_name, cand.code, self.escape_prefix, self.tokenizer
            )
            scored.append((cand.code, enc_cost))
        scored.sort(key=lambda row: (row[1], len(row[0]), row[0]))

        literals: list[tuple[str, float, int, float]] = []
        for item in profile.stats:
            if item.raw_token_cost <= self.assignment_config.min_code_token_cost:
                continue
            weight = self._literal_weight(item.probability, item.raw_token_cost)
            literals.append((item.literal, item.probability, item.raw_token_cost, weight))

        literals.sort(key=lambda row: (-row[3], row[0]))
        literals = literals[:max_assign]

        codes_needed = len(literals)
        if codes_needed == 0:
            return (
                FieldCodebook(
                    field_name=field_name,
                    version=self.version,
                    assignments=[],
                    escape_prefix=self.escape_prefix,
                    metadata={
                        "cost_model": "real_surface_form",
                        "field_name": field_name,
                        "assigned": 0,
                        "feasible_codes": 0,
                    },
                ),
                profile.expected_raw_token_cost,
            )

        checker = PrefixConstraintChecker(
            escape_prefix=self.escape_prefix,
            reserved_strings=reserved,
        )
        feasible_codes: list[tuple[str, int]] = []
        for code, enc_cost in scored:
            if checker.try_add(code):
                feasible_codes.append((code, enc_cost))
            if len(feasible_codes) >= codes_needed:
                break

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

        avg_raw = (
            sum(a.raw_token_cost for a in assignments) / len(assignments) if assignments else 0.0
        )
        avg_enc = (
            sum(a.code_token_cost for a in assignments) / len(assignments) if assignments else 0.0
        )

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
                "cost_model": "real_surface_form",
                "field_name": field_name,
                "avg_raw_token_cost_assigned": avg_raw,
                "avg_encoded_token_cost_assigned": avg_enc,
            },
        )
        return codebook, expected_coded_cost
