"""Run low-cost validation smoke and write JSON artifact."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from config import EVAL_TOKENIZERS
from marker_count import encode as _encode
from repo_miner import _load_tokenizer, collect_py_sources, mine_from_sources
from validate.compare_answers import compare_smoke_payload
from validate.llm_smoke import run_llm_smoke

_ROOT = Path(__file__).resolve().parents[1]
from eval.v2_eval import apply_v2_compression


@contextmanager
def _hybrid_ab_env(
    *,
    stage3_ab_mode: str | None = None,
    enable_b: bool | None = None,
):
    """Temporarily patch hybrid_ab env vars (validate-only; does not change codecs)."""
    saved: dict[str, str | None] = {}
    try:
        if stage3_ab_mode is not None:
            saved["ET_STAGE3_AB_MODE"] = os.environ.get("ET_STAGE3_AB_MODE")
            os.environ["ET_STAGE3_AB_MODE"] = stage3_ab_mode
        if enable_b is not None:
            saved["ET_STAGE3_AB_ENABLE_B"] = os.environ.get("ET_STAGE3_AB_ENABLE_B")
            os.environ["ET_STAGE3_AB_ENABLE_B"] = "1" if enable_b else "0"
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def artifact_filename_for_ab_mode(mode: str) -> str:
    m = (mode or "exact_only").strip().lower()
    if m == "hybrid":
        return "validate_smoke_hybrid_ab_hybrid.json"
    return "validate_smoke_hybrid_ab_exact_only.json"


def run_validate_smoke(
    *,
    tokenizer_key: str = "gpt4",
    max_samples: int = 6,
    stage2_profile: str = "stage2_aggressive",
    stage2_mode: str = "linewise",
    repo_root: Path | None = None,
    stage3_ab_mode: str | None = None,
    enable_b: bool | None = None,
) -> dict:
    env_backend = os.getenv("ET_STAGE3_BACKEND", "").strip().lower()
    if env_backend and env_backend != "hybrid_ab":
        raise ValueError("validate smoke only supports ET_STAGE3_BACKEND=hybrid_ab (or unset).")

    with _hybrid_ab_env(stage3_ab_mode=stage3_ab_mode, enable_b=enable_b):
        cfg = EVAL_TOKENIZERS[tokenizer_key]
        root = repo_root or _ROOT
        sources = collect_py_sources(root)[: max_samples + 2]
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

        ab = getattr(rc, "stage3_ab_summary", {}) or {}
        meta = {
            "stage3_backend": "hybrid_ab",
            "stage3_ab_mode": str(ab.get("mode", "")),
            "enable_b": bool(ab.get("enable_b", False)),
            "tokenizer_key": tokenizer_key,
            "max_samples": max_samples,
            "stage2_profile": stage2_profile,
            "stage2_mode": stage2_mode,
        }

        token_delta = summary["token_delta"]
        token_reduction_pct = summary["token_reduction_pct"]

    return {
        "meta": meta,
        "payload": payload,
        "summary": summary,
        "raw_tokens_total": raw_tokens_total,
        "compressed_tokens_total": compressed_tokens_total,
        "token_delta": token_delta,
        "token_reduction_pct": token_reduction_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="hybrid_ab-only validation smoke.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("ET_VALIDATE_MAX_SAMPLES", "6")),
    )
    parser.add_argument(
        "--stage3-ab-mode",
        choices=("exact_only", "hybrid"),
        default=None,
        help="Override ET_STAGE3_AB_MODE for this run.",
    )
    parser.add_argument(
        "--enable-b",
        type=int,
        choices=(0, 1),
        default=None,
        help="Override ET_STAGE3_AB_ENABLE_B for this run (only meaningful when mode=hybrid).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON here instead of results/artifacts default name.",
    )
    args = parser.parse_args()
    eb = None if args.enable_b is None else bool(args.enable_b)
    out = run_validate_smoke(
        max_samples=max(1, min(args.max_samples, 32)),
        stage3_ab_mode=args.stage3_ab_mode,
        enable_b=eb,
    )
    mode = str(out["meta"].get("stage3_ab_mode", "exact_only"))
    rel = artifact_filename_for_ab_mode(mode)
    out_dir = _ROOT / "results" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = args.output if args.output is not None else out_dir / rel
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(fp)
    print(json.dumps(out["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
