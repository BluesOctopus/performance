#!/usr/bin/env python3
"""
Stage1-only large-corpus evaluation: StarCoder-style Python until 1M raw GPT-4o base tokens.

Does not run Stage2 or Stage3. Token budget uses local tiktoken (gpt-4o / o200k_base).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]

from config import AST_MIN_FREQ, VOCAB_COST_MODE, VOCAB_COST_SCOPE
from lossy_cleaner import lossless_clean
from markers import make_syn_marker
from marker_count import count_augmented, encode as mc_encode
from pipeline import apply_stage1_with_stats
from placeholder_accounting import compute_vocab_intro_cost
from syntax_compressor import (
    SkeletonCandidate,
    build_candidate_pool,
    greedy_mdl_select,
    mine_skeletons,
    build_stage1_vocab_entry,
)
from tokenizer_utils import (
    GPT4oTokenizerResolutionError,
    count_gpt4o_base_tokens,
    resolve_gpt4o_base_tokenizer,
)

log = logging.getLogger("eval_stage1_starcoder_1m")

DEFAULT_CORPUS_JSONL = "stage1_starcoder_1m_corpus.jsonl"
DEFAULT_CHECKPOINT = "stage1_starcoder_1m_checkpoint.json"
DEFAULT_MANIFEST = "stage1_starcoder_1m_manifest.json"


# ---------------------------------------------------------------------------
# HF login & dataset loading
# ---------------------------------------------------------------------------


def login_huggingface_if_needed() -> tuple[bool, list[str]]:
    """Try env-token login; return (logged_in, env_keys_checked)."""
    checked: list[str] = []
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        checked.append(key)
        tok = os.environ.get(key, "").strip()
        if not tok:
            continue
        try:
            from huggingface_hub import login

            login(token=tok, add_to_git_credential=False)
            log.info("Hugging Face login ok via %s", key)
            return True, checked
        except Exception as e:
            log.warning("HF login via %s failed: %s", key, e)
    return False, checked


def load_starcoder_python_stream(
    *,
    dataset_name: str | None,
    language: str = "python",
) -> tuple[Iterator[dict], str, list[dict[str, str]]]:
    """
    Try multiple HF datasets (streaming). Returns (iterator, dataset_used, attempt_log).
    """
    from datasets import load_dataset

    attempt_log: list[dict[str, str]] = []

    candidates: list[str] = []
    if dataset_name:
        candidates.append(dataset_name)
    # Repo default + common StarCoder / stack sources
    from config import EVAL_DATASET

    for name in (
        EVAL_DATASET,
        "bigcode/starcoderdata",
        "bigcode/the-stack",
    ):
        if name not in candidates:
            candidates.append(name)

    lang_lower = language.lower()

    def _row_lang_ok(row: dict, ds_name: str) -> bool:
        if "100star_py" in ds_name or ds_name.endswith("_py"):
            return True
        lang_keys = ("lang", "language", "lang_id", "programming_language")
        if any(row.get(k) is not None for k in lang_keys):
            for k in lang_keys:
                v = row.get(k)
                if v is None:
                    continue
                s = str(v).lower()
                if lang_lower in s or s == lang_lower:
                    return True
            return False
        if "starcoderdata" in ds_name:
            return True
        if ds_name == "bigcode/the-stack":
            return True
        return False

    for ds_name in candidates:
        try:
            kwargs: dict[str, Any] = {"split": "train", "streaming": True}
            if ds_name == "bigcode/the-stack":
                kwargs["data_dir"] = "data/python"

            ds = load_dataset(ds_name, **kwargs)

            def filtered_stream() -> Iterator[dict]:
                for row in ds:
                    if _row_lang_ok(row, ds_name):
                        yield row

            attempt_log.append({"dataset": ds_name, "error": ""})
            log.info("Using dataset stream: %s", ds_name)
            return filtered_stream(), ds_name, attempt_log
        except Exception as e:
            attempt_log.append({"dataset": ds_name, "error": repr(e)})
            log.warning("Dataset %s failed: %s", ds_name, e)

    raise RuntimeError(
        "No StarCoder/Python streaming dataset could be opened. Attempts:\n"
        + json.dumps(attempt_log, indent=2)
    )


# ---------------------------------------------------------------------------
# Text extraction & filtering
# ---------------------------------------------------------------------------

_CODE_TEXT_KEYS = ("content", "code", "text", "body", "source")


def extract_code_text(record: dict) -> str | None:
    for k in _CODE_TEXT_KEYS:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def is_usable_python_sample(text: str, *, min_chars: int = 40) -> bool:
    if not text or len(text.strip()) < min_chars:
        return False
    if len(_NON_PRINTABLE_RE.findall(text)) > max(5, len(text) // 200):
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Corpus collection
# ---------------------------------------------------------------------------


@dataclass
class CollectState:
    accumulated_tokens: int = 0
    num_samples: int = 0
    failed_examples: int = 0
    skipped_examples: int = 0
    skipped_oversized: int = 0
    stream_retries: int = 0
    notes: list[str] = field(default_factory=list)


def collect_starcoder_python_corpus(
    token_budget: int = 1_000_000,
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    corpus_jsonl_path: Path,
    resume: bool = True,
    max_file_base_tokens: int = 200_000,
    dataset_override: str | None = None,
    language: str = "python",
    progress_every_tokens: int = 50_000,
) -> tuple[list[str], dict[str, Any]]:
    """
    Stream HF samples until cumulative **raw** GPT-4o base tokens >= token_budget.
    Persists JSONL + checkpoint for resume.
    """
    login_huggingface_if_needed()

    encoder = resolve_gpt4o_base_tokenizer()
    state = CollectState()
    samples: list[str] = []
    dataset_used = ""
    attempt_log: list[dict[str, str]] = []

    stream_rows_seen = 0
    if resume and corpus_jsonl_path.exists() and checkpoint_path.exists():
        try:
            ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            state.accumulated_tokens = int(ck.get("accumulated_tokens", 0))
            state.num_samples = int(ck.get("num_samples", 0))
            stream_rows_seen = int(ck.get("stream_rows_seen", 0))
            dataset_used = str(ck.get("dataset_used", ""))
            with corpus_jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    samples.append(obj["text"])
            log.info(
                "Resume: loaded %d samples, %d base tokens, stream_rows_seen=%d",
                len(samples),
                state.accumulated_tokens,
                stream_rows_seen,
            )
            if state.accumulated_tokens >= token_budget:
                manifest = _build_manifest(
                    dataset_used=dataset_used or "resumed",
                    token_budget=token_budget,
                    actual_tokens=state.accumulated_tokens,
                    num_files=len(samples),
                    checkpoint_path=checkpoint_path,
                    corpus_path=corpus_jsonl_path,
                    state=state,
                    attempt_log=attempt_log,
                    run_status="complete_from_cache",
                )
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return samples, manifest
        except Exception as e:
            log.warning("Resume failed (%s), starting fresh", e)
            samples = []
            state = CollectState()
            stream_rows_seen = 0

    ds_name_for_stream = dataset_override or (
        dataset_used if dataset_used else None
    )
    stream, dataset_used, attempt_log = load_starcoder_python_stream(
        dataset_name=ds_name_for_stream,
        language=language,
    )

    corpus_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if resume and corpus_jsonl_path.exists() and samples else "w"
    last_progress = state.accumulated_tokens
    row_index = 0

    def _write_ck() -> None:
        checkpoint_path.write_text(
            json.dumps(
                {
                    "dataset_used": dataset_used,
                    "accumulated_tokens": state.accumulated_tokens,
                    "num_samples": state.num_samples,
                    "stream_rows_seen": row_index,
                    "corpus_path": str(corpus_jsonl_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    try:
        with corpus_jsonl_path.open(mode, encoding="utf-8") as outf:
            for row in stream:
                row_index += 1
                if row_index <= stream_rows_seen:
                    continue
                try:
                    text = extract_code_text(row)
                    if text is None or not is_usable_python_sample(text):
                        state.skipped_examples += 1
                        _write_ck()
                        continue
                    ntok = count_gpt4o_base_tokens(text, encoder=encoder)
                    if ntok > max_file_base_tokens:
                        state.skipped_oversized += 1
                        _write_ck()
                        continue
                    rec = {
                        "index": state.num_samples,
                        "text": text,
                        "base_tokens": ntok,
                        "dataset": dataset_used,
                        "source_id": str(row.get("max_stars_repo_name") or row.get("id") or ""),
                    }
                    outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    outf.flush()
                    samples.append(text)
                    state.accumulated_tokens += ntok
                    state.num_samples += 1
                    _write_ck()

                    if state.accumulated_tokens - last_progress >= progress_every_tokens:
                        log.info(
                            "Progress: %d files, %d / %d base tokens",
                            state.num_samples,
                            state.accumulated_tokens,
                            token_budget,
                        )
                        last_progress = state.accumulated_tokens

                    if state.accumulated_tokens >= token_budget:
                        break
                except Exception as e:
                    state.failed_examples += 1
                    log.debug("Skip bad row: %s", e)
                    _write_ck()
    except Exception as e:
        state.notes.append(f"stream_stopped: {e!r}")
        log.exception("Corpus collection interrupted")

    manifest = _build_manifest(
        dataset_used=dataset_used,
        token_budget=token_budget,
        actual_tokens=state.accumulated_tokens,
        num_files=len(samples),
        checkpoint_path=checkpoint_path,
        corpus_path=corpus_jsonl_path,
        state=state,
        attempt_log=attempt_log,
        run_status="complete" if state.accumulated_tokens >= token_budget else "partial",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return samples, manifest


def _build_manifest(
    *,
    dataset_used: str,
    token_budget: int,
    actual_tokens: int,
    num_files: int,
    checkpoint_path: Path,
    corpus_path: Path,
    state: CollectState,
    attempt_log: list[dict[str, str]],
    run_status: str,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_used,
        "token_budget": token_budget,
        "actual_accumulated_base_tokens": actual_tokens,
        "num_files": num_files,
        "checkpoint_path": str(checkpoint_path),
        "corpus_cache_path": str(corpus_path),
        "failed_samples": state.failed_examples,
        "skipped_samples": state.skipped_examples,
        "skipped_oversized": state.skipped_oversized,
        "stream_retries_logged": state.stream_retries,
        "dataset_load_attempts": attempt_log,
        "run_status": run_status,
        "notes": state.notes,
    }


# ---------------------------------------------------------------------------
# Stage1-only run
# ---------------------------------------------------------------------------


@dataclass
class Stage1OnlyRepoConfig:
    """Minimal repo config: only skeleton candidates (no Stage3 map)."""

    selected: list[SkeletonCandidate]

    def skeleton_candidates(self) -> list[SkeletonCandidate]:
        return self.selected

    replacement_map: dict[str, str] = field(default_factory=dict)


def run_stage1_only_large_corpus(
    samples: list[str],
    *,
    tokenizer: Any,
    tok_type: str = "tiktoken",
) -> dict[str, Any]:
    """
    Mine + select Stage1 on corpus; compress each file; aggregate placeholder-aware metrics.
    """
    if not samples:
        return {
            "num_files": 0,
            "baseline_sequence_tokens": 0,
            "stage1_sequence_tokens": 0,
            "stage1_vocab_intro_tokens": 0,
            "stage1_effective_total_tokens": 0,
            "stage1_sequence_reduction_ratio": 0.0,
            "stage1_effective_reduction_ratio": 0.0,
            "selected_skeleton_count": 0,
            "selected_skeletons": [],
            "stage1_vocab_tokens": [],
            "skeleton_rows": [],
        }

    clean_sources: list[str] = []
    for s in samples:
        c, _ = lossless_clean(s)
        clean_sources.append(c)

    N_baseline = 0
    for src in clean_sources:
        N_baseline += len(mc_encode(tokenizer, tok_type, src))

    skeleton_counts = mine_skeletons(clean_sources, min_freq=AST_MIN_FREQ)
    candidates = build_candidate_pool(
        skeleton_counts, tokenizer, tok_type, sources=clean_sources
    )
    V0 = getattr(tokenizer, "n_vocab", None)
    if V0 is None:
        mt = getattr(tokenizer, "max_token_value", None)
        V0 = int(mt) + 1 if mt is not None else 256000
    selected = greedy_mdl_select(candidates, N_baseline, V0)
    cfg = Stage1OnlyRepoConfig(selected=selected)

    baseline_seq = 0
    stage1_seq = 0
    agg_stats: dict[str, dict[str, int]] = {}

    for src in clean_sources:
        baseline_seq += count_augmented(src, tokenizer, tok_type)
        out, stats = apply_stage1_with_stats(src, cfg, tokenizer, tok_type)
        stage1_seq += count_augmented(out, tokenizer, tok_type)
        for sk, st in stats.items():
            a = agg_stats.setdefault(
                sk,
                {
                    "candidate_occurrences": 0,
                    "replaced_occurrences": 0,
                    "skipped_nonpositive_occurrences": 0,
                },
            )
            a["candidate_occurrences"] += int(st.get("candidate_occurrences", 0))
            a["replaced_occurrences"] += int(st.get("replaced_occurrences", 0))
            a["skipped_nonpositive_occurrences"] += int(
                st.get("skipped_nonpositive_occurrences", 0)
            )

    entries = [
        build_stage1_vocab_entry(make_syn_marker(i), c.skeleton)
        for i, c in enumerate(selected)
    ]
    stage1_vocab_intro = compute_vocab_intro_cost(
        entries,
        mode=VOCAB_COST_MODE,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )
    stage1_effective = stage1_seq + stage1_vocab_intro
    base = max(1, baseline_seq)

    skeleton_rows: list[dict[str, Any]] = []
    for i, c in enumerate(selected):
        sk = c.skeleton
        occ = agg_stats.get(sk, {})
        skeleton_rows.append(
            {
                "marker": make_syn_marker(i),
                "skeleton": sk,
                "candidate_occurrences": occ.get("candidate_occurrences", 0),
                "replaced_occurrences": occ.get("replaced_occurrences", 0),
                "skipped_nonpositive_occurrences": occ.get(
                    "skipped_nonpositive_occurrences", 0
                ),
                "total_baseline_sequence_tokens": c.total_baseline_sequence_tokens,
                "total_compressed_sequence_tokens": c.total_compressed_sequence_tokens,
                "total_sequence_net_saving": c.total_net_saving,
                "vocab_intro_tokens": c.vocab_intro_tokens,
                "effective_total_net_saving": c.effective_total_net_saving,
            }
        )

    return {
        "num_files": len(samples),
        "baseline_sequence_tokens": baseline_seq,
        "stage1_sequence_tokens": stage1_seq,
        "stage1_vocab_intro_tokens": stage1_vocab_intro,
        "stage1_effective_total_tokens": stage1_effective,
        "stage1_sequence_reduction_ratio": 1.0 - stage1_seq / base,
        "stage1_effective_reduction_ratio": 1.0 - stage1_effective / base,
        "selected_skeleton_count": len(selected),
        "selected_skeletons": [c.skeleton for c in selected],
        "stage1_vocab_tokens": [make_syn_marker(i) for i in range(len(selected))],
        "skeleton_rows": skeleton_rows,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_stage1_large_corpus_reports(
    result: dict[str, Any],
    *,
    manifest: dict[str, Any],
    output_dir: Path,
    tokenizer_name: str,
    resume_used: bool,
    run_status: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_name": manifest.get("dataset_name", ""),
        "tokenizer_name": tokenizer_name,
        "token_budget": manifest.get("token_budget", 0),
        "actual_accumulated_base_tokens": manifest.get("actual_accumulated_base_tokens", 0),
        "num_files": result["num_files"],
        "baseline_sequence_tokens": result["baseline_sequence_tokens"],
        "stage1_sequence_tokens": result["stage1_sequence_tokens"],
        "stage1_vocab_intro_tokens": result["stage1_vocab_intro_tokens"],
        "stage1_effective_total_tokens": result["stage1_effective_total_tokens"],
        "stage1_sequence_reduction_ratio": result["stage1_sequence_reduction_ratio"],
        "stage1_effective_reduction_ratio": result["stage1_effective_reduction_ratio"],
        "selected_skeleton_count": result["selected_skeleton_count"],
        "vocab_cost_mode": VOCAB_COST_MODE,
        "vocab_cost_scope": VOCAB_COST_SCOPE,
        "run_status": run_status,
        "resume_used": resume_used,
        "notes": (
            "Baseline/stage1 token counts are placeholder-aware (SYN/VAR count as 1). "
            "Corpus size is raw GPT-4o/o200k_base tokens summed until budget."
        ),
    }

    (output_dir / "stage1_starcoder_1m_summary.json").write_text(
        json.dumps({**summary, "manifest": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = output_dir / "stage1_starcoder_1m_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow({k: summary[k] for k in summary})

    sk_path = output_dir / "stage1_starcoder_1m_selected_skeletons.csv"
    rows = result.get("skeleton_rows", [])
    if rows:
        with sk_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        sk_path.write_text(
            "marker,skeleton,candidate_occurrences,replaced_occurrences,"
            "skipped_nonpositive_occurrences,total_baseline_sequence_tokens,"
            "total_compressed_sequence_tokens,total_sequence_net_saving,"
            "vocab_intro_tokens,effective_total_net_saving\n",
            encoding="utf-8",
        )

    vocab_path = output_dir / "stage1_starcoder_1m_vocab_tokens.json"
    vocab_path.write_text(
        json.dumps({"tokens": result.get("stage1_vocab_tokens", [])}, indent=2),
        encoding="utf-8",
    )

    (output_dir / "stage1_starcoder_1m_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Stage1-only StarCoder 1M base-token evaluation.")
    parser.add_argument("--token-budget", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-file-base-tokens", type=int, default=200_000)
    parser.add_argument("--dataset", type=str, default=None, help="Override HF dataset id")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--skip-collect", action="store_true", help="Use existing JSONL only")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    corpus_jsonl = cache_dir / DEFAULT_CORPUS_JSONL
    checkpoint_path = cache_dir / DEFAULT_CHECKPOINT
    manifest_path = cache_dir / DEFAULT_MANIFEST

    resume_used = bool(args.resume)

    try:
        tik = resolve_gpt4o_base_tokenizer()
    except GPT4oTokenizerResolutionError as e:
        log.error("%s", e)
        return 1

    manifest: dict[str, Any] = {}
    run_status = "error"

    if args.skip_collect:
        if not corpus_jsonl.exists():
            log.error("No corpus at %s", corpus_jsonl)
            return 1
        samples = []
        total_tok = 0
        with corpus_jsonl.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                samples.append(o["text"])
                total_tok += int(o.get("base_tokens", 0))
        manifest = {
            "dataset_name": "from_cache_only",
            "token_budget": args.token_budget,
            "actual_accumulated_base_tokens": total_tok,
            "num_files": len(samples),
            "corpus_cache_path": str(corpus_jsonl),
            "run_status": "skip_collect",
        }
        run_status = "skip_collect"
    else:
        try:
            samples, manifest = collect_starcoder_python_corpus(
                args.token_budget,
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                corpus_jsonl_path=corpus_jsonl,
                resume=args.resume,
                max_file_base_tokens=args.max_file_base_tokens,
                dataset_override=args.dataset,
                language=args.language,
            )
            run_status = manifest.get("run_status", "unknown")
        except Exception as e:
            log.exception("Corpus collection failed")
            manifest = {
                "error": repr(e),
                "run_status": "collect_failed",
                "corpus_cache_path": str(corpus_jsonl),
                "checkpoint_path": str(checkpoint_path),
            }
            (args.output_dir).mkdir(parents=True, exist_ok=True)
            (args.output_dir / "stage1_starcoder_1m_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            return 1

    log.info("Running Stage1 on %d files …", len(samples))
    result = run_stage1_only_large_corpus(samples, tokenizer=tik, tok_type="tiktoken")

    write_stage1_large_corpus_reports(
        result,
        manifest=manifest,
        output_dir=args.output_dir,
        tokenizer_name="gpt-4o/o200k_base (tiktoken)",
        resume_used=resume_used,
        run_status=run_status,
    )
    log.info("Wrote reports under %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
