"""Small integer-list summaries for Stage3 hybrid_ab telemetry (no histogram infra)."""

from __future__ import annotations


def int_summary(vals: list[int], *, with_min: bool = True) -> dict[str, float | int]:
    if not vals:
        out: dict[str, float | int] = {"count": 0, "mean": 0.0, "p50": 0, "p90": 0, "max": 0}
        if with_min:
            out["min"] = 0
        return out
    s = sorted(vals)
    n = len(s)

    def _pct(p: float) -> int:
        if n == 1:
            return int(s[0])
        i = int(round((n - 1) * p))
        return int(s[max(0, min(n - 1, i))])

    mean = sum(s) / n
    out2: dict[str, float | int] = {
        "count": n,
        "mean": round(mean, 6),
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "max": int(s[-1]),
    }
    if with_min:
        out2["min"] = int(s[0])
    return out2
