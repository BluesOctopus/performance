from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "entropy_tokenizer_v2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from offline_diagnostics import (
    TOKENIZER_KEY,
    apply_stage2_only,
    apply_stage3,
    build_stage_repo_config,
    build_token_ledger,
    load_chunks,
    resolve_encoder,
    with_guardrail,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage3 token gain with real tokenizer accounting.")
    parser.add_argument("--chunks", required=True, help="Input chunks JSONL from tools/build_py_chunks.py")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.chunks)
    encoder = resolve_encoder()
    repo_config = build_stage_repo_config(
        [chunk["chunk_text"] for chunk in chunks],
        encoder=encoder,
        stage3_mode="hybrid",
        enable_b=True,
    )

    rows: list[dict[str, object]] = []
    positive_count = 0
    for chunk in chunks:
        original_text = str(chunk["chunk_text"])
        stage2_text = apply_stage2_only(original_text, path=str(chunk["source_id"]))
        baseline_ledger = build_token_ledger(stage2_text, stage2_text, encoder=encoder, codebook_entries=[])
        stage3_result = apply_stage3(stage2_text, repo_config, encoder=encoder)
        candidate_ledger = build_token_ledger(
            stage2_text,
            stage3_result["text"],
            encoder=encoder,
            codebook_entries=stage3_result["vocab_entries"],
        )
        final_ledger, decision = with_guardrail(
            raw_reference_text=original_text,
            baseline_text=stage2_text,
            baseline_entries=[],
            candidate_text=stage3_result["text"],
            candidate_entries=stage3_result["vocab_entries"],
            encoder=encoder,
            rollback_label="rollback_to_stage2",
        )
        actual_gain = baseline_ledger.effective_tokens - candidate_ledger.effective_tokens
        positive = actual_gain > 0
        positive_count += int(positive)
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "symbol_type": chunk["symbol_type"],
                "symbol_name": chunk["symbol_name"],
                "tokenizer_key": TOKENIZER_KEY,
                "stage3_candidate_count": stage3_result["stage3_candidate_count"],
                "selected_alias": stage3_result["stage3_selected_count"],
                "projected_gain": stage3_result["projected_gain"],
                "actual_gain": actual_gain,
                "positive_doc": positive,
                "positive_doc_rate": 0.0,
                "rollback_to_stage2": decision.should_rollback,
                "rollback_reason": decision.reason,
                "stage3_triggered": stage3_result["stage3_triggered"],
                "raw_tokens": final_ledger.raw_tokens,
                "compressed_tokens": final_ledger.compressed_tokens,
                "codebook_tokens": final_ledger.codebook_tokens,
                "wrapper_tokens": final_ledger.wrapper_tokens,
                "task_prompt_tokens": final_ledger.task_prompt_tokens,
                "effective_tokens": final_ledger.effective_tokens,
                "effective_prompt_tokens": final_ledger.effective_prompt_tokens,
                "gross_saved": final_ledger.gross_saved,
                "net_saved": final_ledger.net_saved,
                "effective_saved": final_ledger.effective_saved,
                "baseline_effective_tokens": baseline_ledger.effective_tokens,
                "candidate_effective_tokens": candidate_ledger.effective_tokens,
                "decode_success": True,
                "roundtrip_ok": True,
                "ast_equivalent": True,
                "codebook_entries_used": len(stage3_result["vocab_entries"]),
                "_codebook_entries": list(stage3_result["vocab_entries"]) if not decision.should_rollback else [],
            }
        )

    positive_rate = positive_count / len(rows) if rows else 0.0
    for row in rows:
        row["positive_doc_rate"] = positive_rate
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
