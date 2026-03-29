"""Codebook model adapters."""

from __future__ import annotations

from dataclasses import asdict

from literal_codec.types import CodeAssignment, FieldCodebook


def codebook_to_dict(codebook: FieldCodebook) -> dict:
    """Serialize field codebook to dict."""
    return {
        "field_name": codebook.field_name,
        "version": codebook.version,
        "escape_prefix": codebook.escape_prefix,
        "metadata": codebook.metadata,
        "assignments": [asdict(a) for a in codebook.assignments],
    }


def codebook_from_dict(data: dict) -> FieldCodebook:
    """Deserialize field codebook from dict."""
    assignments = [CodeAssignment(**item) for item in data.get("assignments", [])]
    return FieldCodebook(
        field_name=data["field_name"],
        version=data["version"],
        assignments=assignments,
        escape_prefix=data["escape_prefix"],
        metadata=data.get("metadata", {}),
    )
