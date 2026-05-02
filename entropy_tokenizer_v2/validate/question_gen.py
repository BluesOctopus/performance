"""Low-cost question generation for code compression validation smoke."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ValidationQuestion:
    qtype: str
    prompt: str


def generate_questions(source: str) -> list[ValidationQuestion]:
    """Generate 3 fixed question types per sample."""
    head = source.strip().splitlines()
    preview = " ".join(head[:2])[:120]
    return [
        ValidationQuestion(
            qtype="code_qa",
            prompt=f"Summarize what this code does in one short sentence: {preview}",
        ),
        ValidationQuestion(
            qtype="variable_trace",
            prompt="Name one variable that is assigned and later read.",
        ),
        ValidationQuestion(
            qtype="bug_hint",
            prompt="Give one potential bug risk keyword only.",
        ),
    ]

