from __future__ import annotations

import json
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
    markers = list(getattr(repo_config, "stage1_marker_tokens", []) or [])
    entries = []
    for index, stat in enumerate(selected[:top_k]):
        marker = markers[index] if index < len(markers) else f"<SYN_{index}>"
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
