"""Generate low-cost validation questions for code understanding smoke tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ValidationQuestion:
    qid: str
    qtype: str
    question: str


def generate_questions_for_sample(sample_id: str) -> list[ValidationQuestion]:
    """Generate 3 fixed low-cost questions per sample."""
    return [
        ValidationQuestion(
            qid=f"{sample_id}:qa",
            qtype="code_qa",
            question="What does this code primarily do? Answer in one sentence.",
        ),
        ValidationQuestion(
            qid=f"{sample_id}:trace",
            qtype="variable_tracing",
            question="List up to 5 key variables or identifiers involved in the main logic.",
        ),
        ValidationQuestion(
            qid=f"{sample_id}:bug",
            qtype="bug_hint",
            question="Give one plausible bug risk or edge case in this code.",
        ),
    ]
