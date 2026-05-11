from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from offline_diagnostics import build_stage_repo_config, load_chunks
from repo_miner import RepoConfig


def build_static_vocab_manifest(
    chunks: list[dict[str, Any]],
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    top_k: int = 32,
) -> dict[str, Any]:
    repo_config = build_stage_repo_config(
        [str(chunk["chunk_text"]) for chunk in chunks],
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        stage1_marker_scheme="tokenizer_opt",
        stage3_mode="exact_only",
        enable_b=False,
    )
    marker_by_skeleton = {
        str(row.get("skeleton", "")): str(row.get("marker_text", ""))
        for row in list(getattr(repo_config, "selected_skeletons", []) or [])
    }
    selected = [
        stat
        for stat in list(getattr(repo_config, "stage1_selected_stats", []) or [])
        if float(stat.get("avg_sequence_net_saving", 0.0) or 0.0) > 0.0
        and float(stat.get("effective_total_net_saving", 0.0) or 0.0) > 0.0
    ]
    selected.sort(
        key=lambda row: (
            float(row.get("effective_total_net_saving", 0.0) or 0.0),
            float(row.get("avg_sequence_net_saving", 0.0) or 0.0),
            int(row.get("occurrences", 0) or 0),
        ),
        reverse=True,
    )
    entries = []
    for index, stat in enumerate(selected[:top_k]):
        marker = marker_by_skeleton.get(str(stat["skeleton"]), f"<SYN_{index}>")
        entries.append(
            {
                "marker": marker,
                "skeleton": stat["skeleton"],
                "frequency": int(stat.get("occurrences", 0) or 0),
                "avg_token_gain": float(stat.get("avg_sequence_net_saving", 0.0) or 0.0),
                "total_token_gain": int(stat.get("effective_total_net_saving", 0) or 0),
                "decoder_template": stat["skeleton"],
                "safety_status": "statement_template_only",
            }
        )
    return {
        "tokenizer_name": tokenizer_name,
        "top_k": top_k,
        "entry_count": len(entries),
        "entries": entries,
        "source_chunk_count": len(chunks),
    }


def restrict_repo_config_to_static_manifest(
    repo_config: RepoConfig,
    static_manifest: dict[str, Any],
) -> RepoConfig:
    entries = list(static_manifest.get("entries", []) or [])
    if not entries:
        return replace(
            repo_config,
            selected_skeletons=[],
            stage1_selected_stats=[],
            stage1_marker_tokens=[],
            stage1_marker_token_costs=[],
        )

    skeleton_order = {str(entry["skeleton"]): index for index, entry in enumerate(entries)}
    selected_rows = [
        dict(row)
        for row in list(getattr(repo_config, "selected_skeletons", []) or [])
        if str(row.get("skeleton", "")) in skeleton_order
    ]
    selected_rows.sort(key=lambda row: skeleton_order[str(row.get("skeleton", ""))])
    selected_stats = [
        dict(row)
        for row in list(getattr(repo_config, "stage1_selected_stats", []) or [])
        if str(row.get("skeleton", "")) in skeleton_order
    ]
    selected_stats.sort(key=lambda row: skeleton_order[str(row.get("skeleton", ""))])
    marker_cost_by_skeleton = {
        str(row.get("skeleton", "")): int(row.get("marker_cost", 0) or 0)
        for row in selected_rows
    }
    marker_tokens = [str(entry["marker"]) for entry in entries]
    marker_costs = [marker_cost_by_skeleton.get(str(entry["skeleton"]), 0) for entry in entries]
    return replace(
        repo_config,
        selected_skeletons=selected_rows,
        stage1_selected_stats=selected_stats,
        stage1_marker_tokens=marker_tokens,
        stage1_marker_token_costs=marker_costs,
    )


def write_static_vocab_manifest(
    chunks_path: str | Path,
    out_path: str | Path,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    top_k: int = 32,
) -> dict[str, Any]:
    manifest = build_static_vocab_manifest(
        load_chunks(chunks_path),
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
        top_k=top_k,
    )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
