"""Run low-cost validation smoke and write JSON artifact."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from config import EVAL_TOKENIZERS
from marker_count import encode as _encode
from repo_miner import _load_tokenizer, collect_py_sources, mine_from_sources
from validate.compare_answers import compare_smoke_payload
from validate.llm_smoke import run_llm_smoke

_ROOT = Path(__file__).resolve().parents[1]
from eval.v2_eval import apply_v2_compression


def run_validate_smoke(
    *,
    tokenizer_key: str = "gpt4",
    max_samples: int = 6,
    stage2_profile: str = "stage2_aggressive",
    stage2_mode: str = "linewise",
) -> dict:
    cfg = EVAL_TOKENIZERS[tokenizer_key]
    sources = collect_py_sources(".")[: max_samples + 2]
    rc = mine_from_sources(
        sources,
        tokenizer_key=tokenizer_key,
        tokenizer_cfg=cfg,
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend="hybrid_ab",
    )
    tok, tok_type = _load_tokenizer(tokenizer_key, cfg)
    pairs: list[tuple[str, str]] = []
    raw_tokens_total = 0
    compressed_tokens_total = 0
    for src in sources[:max_samples]:
        comp, _ = apply_v2_compression(
            src,
            rc,
            tok,
            tok_type,
            stage2_profile=stage2_profile,
            stage2_mode=stage2_mode,
        )
        pairs.append((src, comp))
        raw_tokens_total += len(_encode(tok, tok_type, src))
        compressed_tokens_total += len(_encode(tok, tok_type, comp))
    payload = run_llm_smoke(pairs, max_samples=max_samples)
    summary = asdict(compare_smoke_payload(payload))
    summary["raw_tokens_total"] = raw_tokens_total
    summary["compressed_tokens_total"] = compressed_tokens_total
    summary["token_delta"] = raw_tokens_total - compressed_tokens_total
    summary["token_reduction_pct"] = (
        (raw_tokens_total - compressed_tokens_total) / raw_tokens_total * 100.0
        if raw_tokens_total
        else 0.0
    )
    return {"payload": payload, "summary": summary}


if __name__ == "__main__":
    out = run_validate_smoke()
    p = Path("results/artifacts")
    p.mkdir(parents=True, exist_ok=True)
    fp = p / "validate_smoke_hybrid_ab_exact_only.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(fp)
    print(json.dumps(out["summary"], ensure_ascii=False))

