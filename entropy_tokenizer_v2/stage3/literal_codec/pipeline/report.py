"""Report generation."""

from __future__ import annotations

from dataclasses import asdict

from literal_codec.types import FieldBuildResult


def field_report(result: FieldBuildResult) -> dict:
    """Generate one-field report dictionary."""
    assignment_by_literal = {a.literal: a for a in result.codebook.assignments}
    top_rows = []
    for item in result.profile.stats:
        assigned = assignment_by_literal.get(item.literal)
        top_rows.append(
            {
                "literal": item.literal,
                "count": item.count,
                "probability": item.probability,
                "surprisal_bits": item.surprisal_bits,
                "raw_token_cost": item.raw_token_cost,
                "assigned_code": assigned.code if assigned else None,
                "code_token_cost": assigned.code_token_cost if assigned else None,
                "expected_gain": assigned.expected_gain if assigned else 0.0,
            }
        )

    return {
        "field_name": result.profile.field_name,
        "sample_size": result.profile.sample_size,
        "cardinality": result.profile.cardinality,
        "entropy_bits": result.profile.entropy_bits,
        "expected_raw_token_cost": result.profile.expected_raw_token_cost,
        "expected_coded_token_cost": result.expected_coded_token_cost,
        "theoretical_headroom": result.theoretical_headroom,
        "dictionary_coverage": result.dictionary_coverage,
        "total_expected_gain": result.total_expected_gain,
        "top_literals": top_rows,
        "codebook": {
            "field_name": result.codebook.field_name,
            "version": result.codebook.version,
            "escape_prefix": result.codebook.escape_prefix,
            "metadata": result.codebook.metadata,
            "assignments": [asdict(a) for a in result.codebook.assignments],
        },
    }


def summary_report(results: list[FieldBuildResult]) -> dict:
    """Generate cross-field summary."""
    total_raw = sum(r.profile.expected_raw_token_cost for r in results)
    total_coded = sum(r.expected_coded_token_cost for r in results)
    total_gain = total_raw - total_coded
    ratio = (total_gain / total_raw) if total_raw > 0 else 0.0
    return {
        "total_expected_raw_tokens": total_raw,
        "total_expected_coded_tokens": total_coded,
        "total_expected_gain": total_gain,
        "average_gain_ratio": ratio,
    }
