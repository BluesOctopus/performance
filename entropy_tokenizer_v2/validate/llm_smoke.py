"""LLM smoke runner for original vs compressed text."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from validate.question_gen import ValidationQuestion, generate_questions

_RE_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


@dataclass(slots=True)
class QAAnswer:
    qtype: str
    question: str
    answer: str


@dataclass(slots=True)
class SampleValidation:
    sample_id: int
    questions: list[ValidationQuestion]
    raw_answers: list[QAAnswer]
    compressed_answers: list[QAAnswer]


def _stub_answer_model(text: str, question: ValidationQuestion) -> str:
    """Cheap deterministic stand-in for strict JSON smoke."""
    tokens = _RE_IDENT.findall(text)
    if question.qtype == "code_qa":
        return f"code touches: {', '.join(tokens[:6]) if tokens else 'none'}"
    if question.qtype == "variable_trace":
        return tokens[0] if tokens else "none"
    if question.qtype == "bug_hint":
        for k in ("timeout", "race", "index", "null", "overflow"):
            if k in text.lower():
                return k
        return "none"
    return "none"


def run_llm_smoke(
    samples: Iterable[tuple[str, str]],
    *,
    max_samples: int = 6,
    llm_fn: Callable[[str, ValidationQuestion], str] | None = None,
) -> dict:
    """Run fixed-question smoke on `(raw, compressed)` pairs."""
    ask = llm_fn or _stub_answer_model
    out: list[SampleValidation] = []
    for idx, (raw, comp) in enumerate(samples):
        if idx >= max_samples:
            break
        qs = generate_questions(raw)
        raw_ans = [QAAnswer(q.qtype, q.prompt, ask(raw, q)) for q in qs]
        comp_ans = [QAAnswer(q.qtype, q.prompt, ask(comp, q)) for q in qs]
        out.append(
            SampleValidation(
                sample_id=idx,
                questions=qs,
                raw_answers=raw_ans,
                compressed_answers=comp_ans,
            )
        )

    payload = {
        "n_samples": len(out),
        "questions_per_sample": 3,
        "samples": [asdict(x) for x in out],
    }
    json.dumps(payload, ensure_ascii=False)
    return payload

