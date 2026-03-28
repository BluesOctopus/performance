"""Offline deterministic ablation runner for the compression pipeline."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import VOCAB_COST_MODE, VOCAB_COST_SCOPE
from markers import make_syn_marker
from pipeline import apply_stage1_with_stats, apply_stage2, apply_stage3
from placeholder_accounting import compute_vocab_intro_cost, count_sequence_tokens
from syntax_compressor import (
    SkeletonCandidate,
    build_candidate_pool,
    greedy_mdl_select,
    mine_skeletons,
    build_stage1_vocab_entry,
)
from token_scorer import (
    build_stage3_vocab_entries_from_used_placeholders,
    collect_used_stage3_placeholders,
)


@dataclass
class ExperimentConfig:
    name: str
    use_stage1: bool
    stage2_profile: str | None
    use_stage3: bool
    notes: str = ""


EXPERIMENTS = [
    ExperimentConfig("baseline", use_stage1=False, stage2_profile=None, use_stage3=False, notes="No compression stages"),
    ExperimentConfig("stage1_only", use_stage1=True, stage2_profile=None, use_stage3=False),
    ExperimentConfig("stage1_stage2_parseable", use_stage1=True, stage2_profile="stage2_parseable", use_stage3=False),
    ExperimentConfig("stage1_stage2_aggressive", use_stage1=True, stage2_profile="stage2_aggressive", use_stage3=False),
    ExperimentConfig("stage1_stage3", use_stage1=True, stage2_profile=None, use_stage3=True),
    ExperimentConfig(
        "stage1_stage2_stage3",
        use_stage1=True,
        stage2_profile="stage2_aggressive",
        use_stage3=True,
        notes="Stage2 uses aggressive profile",
    ),
]


class OfflineTokenizer:
    """Deterministic offline tokenizer that keeps whitespace-sensitive signals."""

    def encode(self, text: str, add_special_tokens: bool = False, allowed_special: str = "all"):
        del add_special_tokens, allowed_special
        out: list[str] = []
        try:
            for tok in tokenize.generate_tokens(StringIO(text).readline):
                if tok.type == tokenize.ENDMARKER:
                    continue
                out.append(f"{tok.type}:{tok.string}")
            return out
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return re.findall(r"[A-Za-z_]\w*|\d+|<[^>\s]+>|\s+|[^\w\s]", text)


class AblationRepoConfig:
    def __init__(self, selected_skeletons: list[SkeletonCandidate]):
        self.replacement_map = {
            "total": "<VAR>",
            "name": "<ATTR>",
            '"alpha"': "<STR>",
            "42": "<NUM>",
            "return_value": "<VAR>",
            '"none"': "<STR>",
            "item": "<VAR>",
            "items": "<VAR>",
            "prompt": "<VAR>",
            '"token"': "<STR>",
            "100": "<NUM>",
        }
        self._skeletons = selected_skeletons
        self.stage1_selected_stats = []
        for i, c in enumerate(self._skeletons):
            self.stage1_selected_stats.append(
                {
                    "marker": make_syn_marker(i),
                    "skeleton": c.skeleton,
                    "occurrences": c.frequency,
                    "avg_net_saving": c.avg_net_saving,
                    "total_net_saving": c.total_net_saving,
                    "vocab_intro_tokens": c.vocab_intro_tokens,
                    "effective_total_net_saving": c.effective_total_net_saving,
                    "total_baseline_sequence_tokens": c.total_baseline_sequence_tokens,
                    "total_compressed_sequence_tokens": c.total_compressed_sequence_tokens,
                    "avg_sequence_net_saving": c.avg_sequence_net_saving,
                    "selected": True,
                }
            )
        self.stage1_total_net_saving = sum(c.effective_total_net_saving for c in self._skeletons)

    def skeleton_candidates(self) -> list[SkeletonCandidate]:
        return self._skeletons


def _load_samples(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_seq(text: str, tokenizer: OfflineTokenizer) -> int:
    return count_sequence_tokens(text, tokenizer=tokenizer, tok_type="hf")


def _is_parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def _corpus_stage1_vocab_intro(
    selected: list[SkeletonCandidate],
    tokenizer: OfflineTokenizer,
) -> int:
    if not selected:
        return 0
    entries = [
        build_stage1_vocab_entry(make_syn_marker(i), c.skeleton)
        for i, c in enumerate(selected)
    ]
    return compute_vocab_intro_cost(
        entries,
        mode=VOCAB_COST_MODE,
        tokenizer=tokenizer,
        tok_type="hf",
    )


def _corpus_stage3_vocab_intro_from_texts(
    texts: list[str],
    rmap: dict[str, str],
    tokenizer: OfflineTokenizer,
) -> tuple[int, list[str]]:
    seen: set[str] = set()
    ordered: list[str] = []
    for t in texts:
        for ph in collect_used_stage3_placeholders(t, rmap):
            if ph not in seen:
                seen.add(ph)
                ordered.append(ph)
    entries = build_stage3_vocab_entries_from_used_placeholders(ordered)
    cost = compute_vocab_intro_cost(
        entries,
        mode=VOCAB_COST_MODE,
        tokenizer=tokenizer,
        tok_type="hf",
    )
    return cost, ordered


def _run_experiment(
    config: ExperimentConfig,
    samples: list[str],
    repo_config: AblationRepoConfig,
    tokenizer: OfflineTokenizer,
) -> tuple[dict[str, str], list[dict[str, str]], list[str]]:
    """Returns summary row (without corpus vocab fill-in), per-file rows, stage3 texts."""
    sum_baseline = 0
    sum_s1 = 0
    sum_s2 = 0
    sum_s3 = 0
    parse_ok = 0
    identity_ok = 0
    per_file_rows: list[dict[str, str]] = []
    stage1_occurrence_stats: dict[str, dict[str, int]] = {}
    stage3_texts: list[str] = []

    for idx, source in enumerate(samples):
        b_seq = _count_seq(source, tokenizer)
        sum_baseline += b_seq

        if config.use_stage1:
            text1, s1_stats = apply_stage1_with_stats(source, repo_config, tokenizer, "hf")
            for skeleton, st in s1_stats.items():
                agg = stage1_occurrence_stats.setdefault(
                    skeleton,
                    {
                        "candidate_occurrences": 0,
                        "replaced_occurrences": 0,
                        "skipped_nonpositive_occurrences": 0,
                        "realized_total_net_saving": 0,
                    },
                )
                agg["candidate_occurrences"] += int(st.get("candidate_occurrences", 0))
                agg["replaced_occurrences"] += int(st.get("replaced_occurrences", 0))
                agg["skipped_nonpositive_occurrences"] += int(
                    st.get("skipped_nonpositive_occurrences", 0)
                )
                agg["realized_total_net_saving"] += int(st.get("total_net_saving_replaced", 0))
        else:
            text1 = source
        s1_seq = _count_seq(text1, tokenizer)
        sum_s1 += s1_seq

        if config.stage2_profile:
            text2 = apply_stage2(text1, profile=config.stage2_profile, mode="linewise")
        else:
            text2 = text1
        s2_seq = _count_seq(text2, tokenizer)
        sum_s2 += s2_seq

        text3 = apply_stage3(text2, repo_config) if config.use_stage3 else text2
        s3_seq = _count_seq(text3, tokenizer)
        sum_s3 += s3_seq
        stage3_texts.append(text3)

        parse_flag = _is_parseable(text3)
        if parse_flag:
            parse_ok += 1

        identity_flag = text3 == source
        if identity_flag:
            identity_ok += 1

        # Per-file: effective_total = sequence only (corpus vocab charged once in summary).
        per_file_rows.append(
            {
                "experiment_name": config.name,
                "file_index": str(idx),
                "original_sequence_tokens": str(b_seq),
                "original_effective_total_tokens": str(b_seq),
                "stage1_sequence_tokens": str(s1_seq),
                "stage1_effective_total_tokens": str(s1_seq),
                "stage2_sequence_tokens": str(s2_seq),
                "stage2_effective_total_tokens": str(s2_seq),
                "stage3_sequence_tokens": str(s3_seq),
                "stage3_effective_total_tokens": str(s3_seq),
                "parse_success": str(parse_flag),
                "identity_preservation": str(identity_flag),
                "changed_stage1": str(text1 != source),
                "changed_stage2": str(text2 != text1),
                "changed_stage3": str(text3 != text2),
                "original_text": json.dumps(source, ensure_ascii=False),
                "stage1_text": json.dumps(text1, ensure_ascii=False),
                "stage2_text": json.dumps(text2, ensure_ascii=False),
                "stage3_text": json.dumps(text3, ensure_ascii=False),
            }
        )

    n = len(samples)
    reduction_ratio = 1.0 - (sum_s3 / sum_baseline) if sum_baseline else 0.0

    summary_row: dict[str, str] = {
        "experiment_name": config.name,
        "num_files": str(n),
        "baseline_sequence_tokens": str(sum_baseline),
        "baseline_vocab_intro_tokens": "0",
        "baseline_effective_total_tokens": str(sum_baseline),
        "stage1_sequence_tokens": str(sum_s1),
        "stage1_vocab_intro_tokens": "0",  # filled after corpus_once
        "stage1_effective_total_tokens": "0",
        "stage2_sequence_tokens": str(sum_s2),
        "stage2_vocab_intro_tokens": "0",
        "stage2_effective_total_tokens": "0",
        "stage3_sequence_tokens": str(sum_s3),
        "stage3_vocab_intro_tokens": "0",
        "stage3_effective_total_tokens": "0",
        "token_count_mode": "sequence_placeholder_aware",
        "vocab_cost_mode": VOCAB_COST_MODE,
        "vocab_cost_scope": VOCAB_COST_SCOPE,
        "stage1_effective_reduction_ratio": "0",
        "stage3_effective_reduction_ratio": "0",
        "compressed_sequence_tokens": str(sum_s3),
        "token_reduction_ratio": f"{reduction_ratio:.6f}",
        "parse_success_rate": f"{parse_ok / n:.6f}" if n else "0.000000",
        "identity_preservation_rate": f"{identity_ok / n:.6f}" if n else "0.000000",
        "notes": config.notes or "",
    }
    summary_row["_stage1_occurrence_stats"] = json.dumps(stage1_occurrence_stats, ensure_ascii=False)
    summary_row["_stage3_texts_json"] = json.dumps(stage3_texts, ensure_ascii=False)
    return summary_row, per_file_rows, stage3_texts


def _finalize_summary_row(
    row: dict[str, str],
    *,
    selected: list[SkeletonCandidate],
    tokenizer: OfflineTokenizer,
    rmap: dict[str, str],
    use_stage1: bool,
    use_stage3: bool,
) -> None:
    """Fill corpus_once vocab columns and effective totals (``VOCAB_COST_SCOPE`` = corpus_once)."""
    sum_baseline = int(row["baseline_sequence_tokens"])
    sum_s1 = int(row["stage1_sequence_tokens"])
    sum_s2 = int(row["stage2_sequence_tokens"])
    sum_s3 = int(row["stage3_sequence_tokens"])

    s1_vocab = _corpus_stage1_vocab_intro(selected, tokenizer) if use_stage1 else 0
    row["stage1_vocab_intro_tokens"] = str(s1_vocab)
    row["stage1_effective_total_tokens"] = str(sum_s1 + s1_vocab)
    row["stage2_vocab_intro_tokens"] = "0"
    row["stage2_effective_total_tokens"] = str(sum_s2)
    if sum_baseline:
        row["stage1_effective_reduction_ratio"] = f"{1.0 - (sum_s1 + s1_vocab) / sum_baseline:.6f}"
    else:
        row["stage1_effective_reduction_ratio"] = "0.000000"

    stage3_texts = json.loads(row.pop("_stage3_texts_json", "[]"))
    s3_vocab, _ = (
        _corpus_stage3_vocab_intro_from_texts(stage3_texts, rmap, tokenizer)
        if use_stage3
        else (0, [])
    )
    row["stage3_vocab_intro_tokens"] = str(s3_vocab)
    row["stage3_effective_total_tokens"] = str(sum_s3 + s3_vocab)
    if sum_baseline:
        row["stage3_effective_reduction_ratio"] = f"{1.0 - (sum_s3 + s3_vocab) / sum_baseline:.6f}"
    else:
        row["stage3_effective_reduction_ratio"] = "0.000000"


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_ablation(
    dataset_path: Path,
    summary_output_path: Path,
    per_file_output_path: Path,
    stage1_selected_output_path: Path,
    stage1_vocab_json_path: Path,
    stage3_vocab_json_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    samples = _load_samples(dataset_path)
    tokenizer = OfflineTokenizer()
    skeleton_counts = mine_skeletons(samples, min_freq=2)
    candidate_pool = build_candidate_pool(skeleton_counts, tokenizer, "hf", samples)
    n_baseline = sum(_count_seq(s, tokenizer) for s in samples)
    selected_skeletons = greedy_mdl_select(candidate_pool, n_baseline, V0=4096)
    repo_config = AblationRepoConfig(selected_skeletons=selected_skeletons)

    summary_rows: list[dict[str, str]] = []
    per_file_rows: list[dict[str, str]] = []
    stage1_selected_occ_stats: dict[str, dict[str, int]] = {}

    for exp in EXPERIMENTS:
        s_row, p_rows, _s3t = _run_experiment(exp, samples, repo_config, tokenizer)
        occ_stats = json.loads(s_row.pop("_stage1_occurrence_stats", "{}"))
        if exp.name == "stage1_only":
            stage1_selected_occ_stats = occ_stats
        _finalize_summary_row(
            s_row,
            selected=selected_skeletons,
            tokenizer=tokenizer,
            rmap=repo_config.replacement_map,
            use_stage1=exp.use_stage1,
            use_stage3=exp.use_stage3,
        )
        selected_stats = getattr(repo_config, "stage1_selected_stats", [])
        s_row["selected_skeleton_count"] = str(len(selected_stats))
        s_row["selected_skeletons"] = "|".join(s["skeleton"] for s in selected_stats)
        s_row["stage1_total_net_saving"] = str(getattr(repo_config, "stage1_total_net_saving", 0))
        summary_rows.append(s_row)
        per_file_rows.extend(p_rows)

    stage1_tokens_list = [make_syn_marker(i) for i in range(len(selected_skeletons))]
    stage1_vocab_json_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_vocab_json_path.write_text(
        json.dumps({"tokens": stage1_tokens_list}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stage3_text_union = [
        json.loads(r["stage3_text"])
        for r in per_file_rows
        if r["experiment_name"] in ("stage1_stage3", "stage1_stage2_stage3")
    ]
    _, stage3_ordered = _corpus_stage3_vocab_intro_from_texts(
        stage3_text_union,
        repo_config.replacement_map,
        tokenizer,
    )
    stage3_vocab_json_path.write_text(
        json.dumps({"tokens": stage3_ordered}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_fields = [
        "experiment_name",
        "num_files",
        "baseline_sequence_tokens",
        "baseline_vocab_intro_tokens",
        "baseline_effective_total_tokens",
        "stage1_sequence_tokens",
        "stage1_vocab_intro_tokens",
        "stage1_effective_total_tokens",
        "stage2_sequence_tokens",
        "stage2_vocab_intro_tokens",
        "stage2_effective_total_tokens",
        "stage3_sequence_tokens",
        "stage3_vocab_intro_tokens",
        "stage3_effective_total_tokens",
        "token_count_mode",
        "vocab_cost_mode",
        "vocab_cost_scope",
        "stage1_effective_reduction_ratio",
        "stage3_effective_reduction_ratio",
        "compressed_sequence_tokens",
        "token_reduction_ratio",
        "parse_success_rate",
        "identity_preservation_rate",
        "selected_skeleton_count",
        "selected_skeletons",
        "stage1_total_net_saving",
        "notes",
    ]
    per_file_fields = [
        "experiment_name",
        "file_index",
        "original_sequence_tokens",
        "original_effective_total_tokens",
        "stage1_sequence_tokens",
        "stage1_effective_total_tokens",
        "stage2_sequence_tokens",
        "stage2_effective_total_tokens",
        "stage3_sequence_tokens",
        "stage3_effective_total_tokens",
        "parse_success",
        "identity_preservation",
        "changed_stage1",
        "changed_stage2",
        "changed_stage3",
        "original_text",
        "stage1_text",
        "stage2_text",
        "stage3_text",
    ]
    _write_csv(summary_output_path, summary_rows, summary_fields)
    _write_csv(per_file_output_path, per_file_rows, per_file_fields)

    stage1_selected_rows = []
    for i, s in enumerate(getattr(repo_config, "stage1_selected_stats", [])):
        sk = s["skeleton"]
        occ = stage1_selected_occ_stats.get(sk, {})
        c = selected_skeletons[i] if i < len(selected_skeletons) else None
        stage1_selected_rows.append(
            {
                "marker": s.get("marker", make_syn_marker(i)),
                "skeleton": sk,
                "candidate_occurrences": str(occ.get("candidate_occurrences", 0)),
                "replaced_occurrences": str(occ.get("replaced_occurrences", 0)),
                "skipped_nonpositive_occurrences": str(
                    occ.get("skipped_nonpositive_occurrences", 0)
                ),
                "total_baseline_sequence_tokens": str(
                    getattr(c, "total_baseline_sequence_tokens", 0) if c else 0
                ),
                "total_compressed_sequence_tokens": str(
                    getattr(c, "total_compressed_sequence_tokens", 0) if c else 0
                ),
                "total_sequence_net_saving": str(getattr(c, "total_net_saving", 0) if c else 0),
                "vocab_intro_tokens": str(getattr(c, "vocab_intro_tokens", 0) if c else 0),
                "effective_total_net_saving": str(
                    getattr(c, "effective_total_net_saving", 0) if c else 0
                ),
                "occurrences": str(s.get("occurrences", 0)),
                "avg_net_saving": str(s.get("avg_net_saving", 0)),
                "total_net_saving_legacy": str(s.get("total_net_saving", 0)),
            }
        )

    _write_csv(
        stage1_selected_output_path,
        stage1_selected_rows,
        [
            "marker",
            "skeleton",
            "candidate_occurrences",
            "replaced_occurrences",
            "skipped_nonpositive_occurrences",
            "total_baseline_sequence_tokens",
            "total_compressed_sequence_tokens",
            "total_sequence_net_saving",
            "vocab_intro_tokens",
            "effective_total_net_saving",
            "occurrences",
            "avg_net_saving",
            "total_net_saving_legacy",
        ],
    )
    return summary_rows, per_file_rows, stage1_selected_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic offline ablation.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/data/offline_ablation_samples.json"),
        help="Offline sample dataset path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/ablation_summary.csv"),
        help="Summary CSV output path",
    )
    parser.add_argument(
        "--per-file-output",
        type=Path,
        default=Path("results/per_file_ablation.csv"),
        help="Per-file CSV output path",
    )
    parser.add_argument(
        "--stage1-selected-output",
        type=Path,
        default=Path("results/stage1_selected_skeletons.csv"),
        help="Stage1 selected skeleton stats output path",
    )
    parser.add_argument(
        "--stage1-vocab-json",
        type=Path,
        default=Path("results/stage1_vocab_tokens.json"),
        help="Stage1 selected SYN markers JSON",
    )
    parser.add_argument(
        "--stage3-vocab-json",
        type=Path,
        default=Path("results/stage3_vocab_tokens.json"),
        help="Stage3 used placeholders JSON (corpus union for primary experiment)",
    )
    args = parser.parse_args()

    summary_rows, per_file_rows, stage1_selected_rows = run_ablation(
        args.dataset,
        args.summary_output,
        args.per_file_output,
        args.stage1_selected_output,
        args.stage1_vocab_json,
        args.stage3_vocab_json,
    )
    print(f"[ablation] experiments={len(summary_rows)}")
    print(f"[ablation] per_file_rows={len(per_file_rows)}")
    print(f"[ablation] stage1_selected_rows={len(stage1_selected_rows)}")
    print(f"[ablation] dataset={args.dataset.resolve()}")
    print(f"[ablation] summary_output={args.summary_output.resolve()}")
    print(f"[ablation] per_file_output={args.per_file_output.resolve()}")
    print(f"[ablation] stage1_selected_output={args.stage1_selected_output.resolve()}")
    print(f"[ablation] stage1_vocab_json={args.stage1_vocab_json.resolve()}")
    print(f"[ablation] stage3_vocab_json={args.stage3_vocab_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
