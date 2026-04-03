"""Validation smoke runner with strict JSON answer contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from validate.question_gen import ValidationQuestion, generate_questions_for_sample


@dataclass(slots=True)
class SmokeSample:
    sample_id: str
    original_text: str
    compressed_text: str


def _default_answer_fn(text: str, q: ValidationQuestion) -> str:
    """
    Cheap local baseline responder.
    Returns JSON string with a single 'answer' field.
    """
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    top = words[:10]
    if q.qtype == "variable_tracing":
        ans = ", ".join(top[:5])
    elif q.qtype == "bug_hint":
        ans = "Potential edge case: empty input, None value, or index boundary handling."
    else:
        ans = f"This code appears to handle {top[0] if top else 'processing'} logic."
    return json.dumps({"answer": ans}, ensure_ascii=False)


def _parse_answer(raw: str) -> str:
    obj = json.loads(raw)
    if not isinstance(obj, dict) or "answer" not in obj:
        raise ValueError("Invalid answer JSON schema")
    ans = obj["answer"]
    if not isinstance(ans, str):
        raise ValueError("Answer must be string")
    return ans


def run_validation_smoke(
    samples: list[SmokeSample],
    *,
    answer_fn: Callable[[str, ValidationQuestion], str] | None = None,
) -> dict:
    """
    Run original-vs-compressed QA smoke.

    Returns structured JSON-friendly dict for compare_answers.
    """
    afn = answer_fn or _default_answer_fn
    out: list[dict] = []
    for s in samples:
        qs = generate_questions_for_sample(s.sample_id)
        qrows: list[dict] = []
        for q in qs:
            o_raw = afn(s.original_text, q)
            c_raw = afn(s.compressed_text, q)
            qrows.append(
                {
                    "qid": q.qid,
                    "qtype": q.qtype,
                    "question": q.question,
                    "original_answer": _parse_answer(o_raw),
                    "compressed_answer": _parse_answer(c_raw),
                }
            )
        out.append({"sample_id": s.sample_id, "qa": qrows})
    return {"samples": out, "n_samples": len(samples), "questions_per_sample": 3}
