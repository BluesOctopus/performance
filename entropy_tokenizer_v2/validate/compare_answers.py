"""Lightweight automatic comparison for validation smoke outputs."""

from __future__ import annotations

import re


def _tokenize(s: str) -> set[str]:
    return {x.lower() for x in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)}


def _overlap(a: str, b: str) -> float:
    sa = _tokenize(a)
    sb = _tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def compare_smoke_answers(payload: dict) -> dict:
    rows = payload.get("samples", [])
    scores: list[float] = []
    by_type: dict[str, list[float]] = {"code_qa": [], "variable_tracing": [], "bug_hint": []}
    for s in rows:
        for q in s.get("qa", []):
            ov = _overlap(q.get("original_answer", ""), q.get("compressed_answer", ""))
            scores.append(ov)
            by_type.setdefault(q.get("qtype", "unknown"), []).append(ov)
    mean = sum(scores) / len(scores) if scores else 0.0
    per_type = {k: (sum(v) / len(v) if v else 0.0) for k, v in by_type.items()}
    rollback = mean < 0.55
    return {
        "overall_overlap": mean,
        "by_type_overlap": per_type,
        "rollback_suggested": rollback,
        "rollback_reason": "answer overlap too low" if rollback else "",
    }
