"""CLI helper: run low-cost validation smoke on local sources."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from config import EVAL_TOKENIZERS
from repo_miner import _load_tokenizer, collect_py_sources, mine_from_sources
from validate.compare_answers import compare_smoke_answers
from validate.llm_smoke import SmokeSample, run_validation_smoke

_ROOT = Path(__file__).resolve().parents[1]
_EVAL = _ROOT / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))
import bootstrap_v2

bootstrap_v2.ensure()

from v2_eval import apply_v2_compression


def run_validate_smoke(
    *,
    n_samples: int = 5,
    tokenizer_key: str = "gpt4",
    stage3_backend: str = "hybrid_ab",
    output_name: str = "validate_smoke.json",
) -> dict:
    sources = collect_py_sources(".")[: max(1, n_samples)]
    tok_cfg = EVAL_TOKENIZERS[tokenizer_key]
    rc = mine_from_sources(
        sources,
        tokenizer_key=tokenizer_key,
        tokenizer_cfg=tok_cfg,
        cache=False,
        verbose=False,
        min_freq=1,
        stage3_backend=stage3_backend,
    )
    tok, tt = _load_tokenizer(tokenizer_key, tok_cfg)
    samples: list[SmokeSample] = []
    for i, src in enumerate(sources):
        compressed, _ = apply_v2_compression(
            src,
            rc,
            tok,
            tt,
            stage2_profile="stage2_aggressive",
            stage2_mode="linewise",
        )
        samples.append(
            SmokeSample(sample_id=f"s{i}", original_text=src, compressed_text=compressed)
        )
    payload = run_validation_smoke(samples)
    compare = compare_smoke_answers(payload)
    out = {"payload": payload, "compare": compare}
    out_path = Path("results") / "artifacts" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_path": str(out_path), **compare}


if __name__ == "__main__":
    print(json.dumps(run_validate_smoke(), ensure_ascii=False))
