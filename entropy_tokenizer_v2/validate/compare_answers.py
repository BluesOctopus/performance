"""Lightweight scoring for smoke outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


def _tok(s: str) -> list[str]:
    return [x.lower() for x in s.replace(",", " ").replace(".", " ").split() if x.strip()]


def _f1(a: Iterable[str], b: Iterable[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    p = common / max(1, sum(cb.values()))
    r = common / max(1, sum(ca.values()))
    return 2 * p * r / max(1e-9, p + r)


@dataclass(slots=True)
class CompareSummary:
    qa_overlap: float
    trace_f1: float
    bug_keyword_overlap: float
    rollback_recommended: bool


def compare_smoke_payload(payload: dict) -> CompareSummary:
    qa_scores: list[float] = []
    trace_scores: list[float] = []
    bug_scores: list[float] = []
    for sample in payload.get("samples", []):
        raws = {x["qtype"]: x["answer"] for x in sample.get("raw_answers", [])}
        comps = {x["qtype"]: x["answer"] for x in sample.get("compressed_answers", [])}
        qa_scores.append(_f1(_tok(raws.get("code_qa", "")), _tok(comps.get("code_qa", ""))))
        trace_scores.append(_f1(_tok(raws.get("variable_trace", "")), _tok(comps.get("variable_trace", ""))))
        bug_scores.append(_f1(_tok(raws.get("bug_hint", "")), _tok(comps.get("bug_hint", ""))))

    qa = sum(qa_scores) / len(qa_scores) if qa_scores else 0.0
    tr = sum(trace_scores) / len(trace_scores) if trace_scores else 0.0
    bg = sum(bug_scores) / len(bug_scores) if bug_scores else 0.0
    return CompareSummary(
        qa_overlap=qa,
        trace_f1=tr,
        bug_keyword_overlap=bg,
        rollback_recommended=(qa < 0.4 or tr < 0.4),
    )

