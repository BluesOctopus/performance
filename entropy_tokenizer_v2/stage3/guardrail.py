from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GuardrailDecision:
    baseline_effective_tokens: int
    candidate_effective_tokens: int
    should_rollback: bool
    rollback_label: str
    reason: str

    def to_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


def apply_effective_guardrail(
    *,
    baseline_effective_tokens: int,
    candidate_effective_tokens: int,
    rollback_label: str,
) -> GuardrailDecision:
    should_rollback = int(candidate_effective_tokens) >= int(baseline_effective_tokens)
    reason = (
        f"{rollback_label}: candidate_effective_tokens >= baseline_effective_tokens"
        if should_rollback
        else ""
    )
    return GuardrailDecision(
        baseline_effective_tokens=int(baseline_effective_tokens),
        candidate_effective_tokens=int(candidate_effective_tokens),
        should_rollback=should_rollback,
        rollback_label=rollback_label,
        reason=reason,
    )
