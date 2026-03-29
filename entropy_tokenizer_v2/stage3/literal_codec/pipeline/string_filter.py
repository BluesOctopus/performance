"""Heuristic filters for Plan A string literals (gpt4 conservative paths)."""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StringFilterConfig:
    """Per-corpus string literal gate before FieldProfiler."""

    min_count: int = 1
    min_raw_token_cost: int = 0
    strict_heuristics: bool = False


@dataclass
class StringFilterDiagnostics:
    """Counts of dropped string *occurrences* by rule."""

    kept_occurrences: int = 0
    dropped_low_count: int = 0
    dropped_low_raw_cost: int = 0
    dropped_newline_inner: int = 0
    dropped_long_prose: int = 0
    dropped_regex_like: int = 0
    dropped_doc_marker: int = 0
    dropped_high_whitespace_ratio: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage3_plan_a_string_filter_kept_occurrences": self.kept_occurrences,
            "stage3_plan_a_string_filter_dropped_low_count": self.dropped_low_count,
            "stage3_plan_a_string_filter_dropped_low_raw_cost": self.dropped_low_raw_cost,
            "stage3_plan_a_string_filter_dropped_newline_inner": self.dropped_newline_inner,
            "stage3_plan_a_string_filter_dropped_long_prose": self.dropped_long_prose,
            "stage3_plan_a_string_filter_dropped_regex_like": self.dropped_regex_like,
            "stage3_plan_a_string_filter_dropped_doc_marker": self.dropped_doc_marker,
            "stage3_plan_a_string_filter_dropped_high_whitespace_ratio": self.dropped_high_whitespace_ratio,
            "stage3_plan_a_string_filter_by_rule_json": self.by_rule,
        }


_RE_DOC_MARKERS = re.compile(
    r"(Stage\s*[123]|EffTotal|Vocab\s*=|sequence_reduction|effective_total|Copyright \(c\))",
    re.I,
)
_RE_REGEXY = re.compile(
    r"(\\b|\\[A-Za-z_\]\+|\[\^\\w\\s\]|\\d\+|\\s\*|\(\?:|\\\\[nrt])"
)


def _inner_from_string_token(spelling: str) -> str | None:
    try:
        v = ast.literal_eval(spelling)
    except (SyntaxError, ValueError, MemoryError):
        return None
    return v if isinstance(v, str) else None


def string_literal_should_drop(
    spelling: str,
    *,
    inner: str,
    count: int,
    raw_token_cost: int,
    cfg: StringFilterConfig,
) -> tuple[bool, str | None]:
    """Return (drop, reason_key)."""
    if count < cfg.min_count:
        return True, "low_count"
    if raw_token_cost < cfg.min_raw_token_cost:
        return True, "low_raw_cost"
    if not cfg.strict_heuristics:
        return False, None
    if "\n" in inner or "\r" in inner:
        return True, "inner_newline"
    if len(inner) > 1200:
        return True, "long_prose"
    if _RE_DOC_MARKERS.search(inner):
        return True, "doc_marker"
    if len(inner) >= 24 and _RE_REGEXY.search(inner):
        return True, "regex_like"
    if len(inner) >= 80:
        nspace = inner.count(" ")
        ratio = nspace / max(len(inner), 1)
        if ratio > 0.38 and len(inner.split()) > 12:
            return True, "high_whitespace_ratio"
    return False, None


def filter_string_occurrences(
    occurrences: list[str],
    raw_cost_fn: Callable[[str], int],
    cfg: StringFilterConfig,
) -> tuple[list[str], StringFilterDiagnostics]:
    """Filter STRING token spellings before frequency profiling."""
    diag = StringFilterDiagnostics()
    if not occurrences:
        return [], diag

    cnt = Counter(occurrences)
    keep_spell: dict[str, bool] = {}
    reason_by_spell: dict[str, str | None] = {}

    for sp in cnt:
        inner = _inner_from_string_token(sp)
        if inner is None:
            keep_spell[sp] = False
            reason_by_spell[sp] = "bad_literal_eval"
            continue
        rtc = int(raw_cost_fn(sp))
        drop, reason = string_literal_should_drop(
            sp, inner=inner, count=cnt[sp], raw_token_cost=rtc, cfg=cfg
        )
        keep_spell[sp] = not drop
        reason_by_spell[sp] = reason if drop else None

    out: list[str] = []
    br: dict[str, int] = {}
    for sp in occurrences:
        if keep_spell.get(sp, False):
            out.append(sp)
        else:
            r = reason_by_spell.get(sp) or "unknown"
            br[r] = br.get(r, 0) + 1
            if r == "low_count":
                diag.dropped_low_count += 1
            elif r == "low_raw_cost":
                diag.dropped_low_raw_cost += 1
            elif r == "inner_newline":
                diag.dropped_newline_inner += 1
            elif r == "long_prose":
                diag.dropped_long_prose += 1
            elif r == "regex_like":
                diag.dropped_regex_like += 1
            elif r == "doc_marker":
                diag.dropped_doc_marker += 1
            elif r == "high_whitespace_ratio":
                diag.dropped_high_whitespace_ratio += 1

    diag.kept_occurrences = len(out)
    diag.by_rule = br
    return out, diag
