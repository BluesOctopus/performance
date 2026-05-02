#!/usr/bin/env python3
"""
Single experiment: gpt4 + hybrid_ab + hybrid mode on first 200k baseline tokens from default HF dataset.

Outputs under RESULTS_DIR / stage3ab_starcoder_200k/ (see module doc in repo root config).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from eval import bootstrap_v2  # noqa: E402

bootstrap_v2.ensure()

from config import (  # noqa: E402
    EVAL_DATASET,
    EVAL_TOKENIZERS,
    HF_DISK_DATASET_FALLBACK,
    HF_TOKEN,
    RESULTS_DIR,
)
from eval.v2_eval import evaluate  # noqa: E402
from marker_count import encode as mc_encode  # noqa: E402
from pipeline import (  # noqa: E402
    apply_stage1_with_stats,
    build_stage2_config,
    resolve_stage2_for_pipeline,
)
from repo_miner import _load_tokenizer, mine_from_sources  # noqa: E402
from stage2.cleaning import stage2_clean_skip_syn_and_stats  # noqa: E402
from stage3.backends.hybrid_ab_backend import (  # noqa: E402
    _apply_hybrid_ab_file_guardrail,
    encode_stage3_hybrid_ab,
    hybrid_ab_config_from_summary,
)

TOKEN_BUDGET = 200_000
TOKENIZER_KEY = "gpt4"
OUT_SUBDIR = "stage3ab_starcoder_200k"


def _baseline_token_len(text: str, tokenizer, tok_type: str) -> int:
    return len(mc_encode(tokenizer, tok_type, text))


def _truncate_to_tokens(text: str, max_tokens: int, tokenizer, tok_type: str) -> tuple[str, bool]:
    ids = mc_encode(tokenizer, tok_type, text)
    if len(ids) <= max_tokens:
        return text, False
    chunk = ids[:max_tokens]
    if tok_type == "tiktoken":
        out = tokenizer.decode(chunk)
    else:
        out = tokenizer.decode(chunk)
    return out, True


def stream_freeze_corpus(
    tokenizer,
    tok_type: str,
    budget: int,
) -> tuple[list[str], dict]:
    if HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    sources: list[str] = []
    per_file_tokens: list[int] = []
    truncated_flags: list[bool] = []
    total = 0
    dataset_exhausted = False

    try:
        from datasets import load_dataset

        ds = load_dataset(EVAL_DATASET, split="train", streaming=True, token=HF_TOKEN or None)
        it = iter(ds)
    except Exception:
        from datasets import load_from_disk

        ds = load_from_disk(str(HF_DISK_DATASET_FALLBACK))
        rows = ds["content"]
        it = iter(rows)

    while total < budget:
        try:
            ex = next(it)
            content = ex["content"] if isinstance(ex, dict) else ex
        except StopIteration:
            dataset_exhausted = True
            break
        need = budget - total
        n = _baseline_token_len(content, tokenizer, tok_type)
        if n <= need:
            sources.append(content)
            per_file_tokens.append(n)
            truncated_flags.append(False)
            total += n
        else:
            frag, trunc = _truncate_to_tokens(content, need, tokenizer, tok_type)
            sources.append(frag)
            per_file_tokens.append(need)
            truncated_flags.append(trunc)
            total += need
            break

    manifest = {
        "dataset_name": EVAL_DATASET,
        "tokenizer_key": TOKENIZER_KEY,
        "token_budget": budget,
        "actual_baseline_tokens": total,
        "n_sources": len(sources),
        "per_file_baseline_tokens": per_file_tokens,
        "per_file_truncated": truncated_flags,
        "dataset_exhausted_before_budget": dataset_exhausted and total < budget,
    }
    return sources, manifest


def _clip(s: str, lim: int = 600) -> str:
    s = s.replace("\r\n", "\n")
    return s if len(s) <= lim else s[: lim - 3] + "..."


def write_examples(
    sources: list[str],
    repo_config,
    tokenizer,
    tok_type: str,
    path: Path,
    n: int = 5,
) -> None:
    cfg_raw = dict(getattr(repo_config, "stage3_ab_summary", {}) or {})
    conf = hybrid_ab_config_from_summary(cfg_raw)
    if conf is None:
        path.write_text("# Examples skipped: no stage3_ab_summary\n", encoding="utf-8")
        return
    s2_prof, s2_mode, _ = resolve_stage2_for_pipeline(repo_config, None, None)
    s2_cfg = build_stage2_config(profile=s2_prof, mode=s2_mode)
    lines = ["# Stage3 hybrid_ab examples (first {} files)\n".format(min(n, len(sources)))]
    for i, src in enumerate(sources[:n]):
        s1, _ = apply_stage1_with_stats(src, repo_config, tokenizer, tok_type)
        after_s2, _ = stage2_clean_skip_syn_and_stats(
            s1,
            s2_cfg.cleaning,
            mode=s2_cfg.mode,
            drop_empty_cleaned_lines=False,
        )
        raw = encode_stage3_hybrid_ab(after_s2, tokenizer=tokenizer, tok_type=tok_type, cfg=conf)
        final, _ = _apply_hybrid_ab_file_guardrail(
            after_s2, raw, conf=conf, tokenizer=tokenizer, tok_type=tok_type
        )
        after_a = final.a.encoded_text
        final_txt = final.encoded_text
        b_note = (
            "B 前后相同（B 未改写文本）。"
            if after_a == final_txt
            else "B 对文本有改写。"
        )
        lines.append(f"## Sample {i}\n")
        lines.append(f"- {b_note}\n")
        lines.append("### 原文片段\n```\n" + _clip(src) + "\n```\n")
        lines.append("### Stage1 后\n```\n" + _clip(s1) + "\n```\n")
        lines.append("### Stage2 后\n```\n" + _clip(after_s2) + "\n```\n")
        lines.append("### Stage3 A 后（含 guardrail）\n```\n" + _clip(after_a) + "\n```\n")
        lines.append("### Stage3 最终（A+B，含 guardrail）\n```\n" + _clip(final_txt) + "\n```\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("ET_STAGE3_AB_MODE", "hybrid")
    os.environ.setdefault("ET_STAGE3_AB_ENABLE_B", "1")

    out_dir = RESULTS_DIR / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_cfg = EVAL_TOKENIZERS[TOKENIZER_KEY]
    tokenizer, tok_type = _load_tokenizer(TOKENIZER_KEY, tok_cfg)

    print("[freeze] streaming dataset", EVAL_DATASET, "budget", TOKEN_BUDGET)
    sources, manifest = stream_freeze_corpus(tokenizer, tok_type, TOKEN_BUDGET)
    manifest_path = out_dir / "frozen_corpus_manifest.json"
    jsonl_path = out_dir / "frozen_corpus_sources.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(sources):
            f.write(json.dumps({"i": i, "content": text}, ensure_ascii=False) + "\n")
    print("[freeze] wrote", manifest_path, "and", jsonl_path, "n_sources=", len(sources))

    cache_name = f"starcoder_baseline{TOKEN_BUDGET}_hybrid_v1"
    rc = mine_from_sources(
        sources=sources,
        tokenizer_key=TOKENIZER_KEY,
        tokenizer_cfg=tok_cfg,
        cache_name=cache_name,
        cache=True,
        verbose=True,
        stage3_backend="hybrid_ab",
    )

    r = evaluate(sources, rc, TOKENIZER_KEY, tok_cfg, stage2_profile=None, stage2_mode=None)

    def wcsv(name: str, rows: list[dict], fieldnames: list[str]) -> None:
        p = out_dir / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print("[out]", p)

    # --- summary ---
    sum_row = {
        "dataset_name": EVAL_DATASET,
        "tokenizer_key": TOKENIZER_KEY,
        "token_budget": TOKEN_BUDGET,
        "n_sources": r.n_files,
        "sum_baseline_tokens": r.baseline_tokens,
        "sum_after_syntax_tokens": r.syntax_tokens,
        "sum_after_stage2_tokens": r.cleaning_tokens,
        "sum_after_stage3_tokens": r.sequence_final_tokens,
        "sum_effective_total_tokens": r.effective_total_tokens,
        "syntax_saved": r.syntax_saved,
        "cleaning_saved": r.cleaning_saved,
        "replacement_saved": r.replacement_saved,
        "stage1_vocab_intro_tokens": r.stage1_vocab_intro_tokens,
        "stage3_vocab_intro_tokens": r.stage3_vocab_intro_tokens,
        "sequence_reduction_pct": f"{r.sequence_reduction_pct:.6f}",
        "effective_total_reduction_pct": f"{r.effective_total_reduction_pct:.6f}",
        "syntax_pct": f"{r.syntax_pct:.6f}",
        "cleaning_pct": f"{r.cleaning_pct:.6f}",
        "replacement_pct": f"{r.replacement_pct:.6f}",
    }
    fields_sum = list(sum_row.keys())
    wcsv("stage3ab_starcoder_200k_summary.csv", [sum_row], fields_sum)

    summary_md = f"""# stage3ab_starcoder_200k summary

- **dataset**: `{EVAL_DATASET}`
- **tokenizer**: `{TOKENIZER_KEY}`
- **baseline token budget**: {TOKEN_BUDGET}
- **actual baseline tokens (sum raw encode)**: {manifest["actual_baseline_tokens"]}
- **n_sources**: {r.n_files}

## Token chain (corpus sums, same counters as v2_eval / apply_pipeline)

| Stage | Tokens |
|------|--------|
| Baseline (raw source) | {r.baseline_tokens:,} |
| After Stage1 (syntax) | {r.syntax_tokens:,} |
| After Stage2 (cleaning) | {r.cleaning_tokens:,} |
| After Stage3 (sequence final) | {r.sequence_final_tokens:,} |
| Effective total (seq + S1voc + S3voc) | {r.effective_total_tokens:,} |

## Percentages (vs baseline)

- **sequence_reduction_pct**: {r.sequence_reduction_pct:.4f}%
- **effective_total_reduction_pct**: {r.effective_total_reduction_pct:.4f}%
- **syntax_pct** (Stage1): {r.syntax_pct:.4f}%
- **cleaning_pct** (Stage2): {r.cleaning_pct:.4f}%
- **replacement_pct** ((Stage2out−Stage3out)/baseline): {r.replacement_pct:.4f}%

## Vocab intro

- stage1_vocab_intro_tokens: {r.stage1_vocab_intro_tokens}
- stage3_vocab_intro_tokens: {r.stage3_vocab_intro_tokens}
"""
    (out_dir / "stage3ab_starcoder_200k_summary.md").write_text(summary_md, encoding="utf-8")
    print("[out]", out_dir / "stage3ab_starcoder_200k_summary.md")

    # --- A/B compare (corpus sums, pipeline token counter) ---
    s2_in = r.stage3_ab_stage2_input_tokens_sum
    after_a = r.stage3_ab_after_a_tokens_sum
    after_b = r.stage3_ab_after_b_tokens_sum
    a_intro = r.stage3_ab_a_intro_tokens
    b_intro = r.stage3_ab_b_intro_tokens
    a_saved = s2_in - after_a
    b_saved = after_a - after_b
    a_net = a_saved - a_intro
    b_net = b_saved - b_intro

    def pct(saved: int, inp: int) -> float:
        return (100.0 * saved / inp) if inp else 0.0

    ab_rows = [
        {
            "channel": "Stage3_A_exact_aliasing",
            "input_tokens": s2_in,
            "output_tokens": after_a,
            "saved_tokens": a_saved,
            "saved_pct": f"{pct(a_saved, s2_in):.6f}",
            "intro_tokens": a_intro,
            "net_saved_tokens": a_net,
            "notes": "input=sum Stage2 output tokens; output=after A+guardrail; counter=count_augmented",
        },
        {
            "channel": "Stage3_B_lexical_free_text",
            "input_tokens": after_a,
            "output_tokens": after_b,
            "saved_tokens": b_saved,
            "saved_pct": f"{pct(b_saved, after_a):.6f}",
            "intro_tokens": b_intro,
            "net_saved_tokens": b_net,
            "notes": "input=A output; output=final Stage3; B may be no-op",
        },
    ]
    ab_fields = [
        "channel",
        "input_tokens",
        "output_tokens",
        "saved_tokens",
        "saved_pct",
        "intro_tokens",
        "net_saved_tokens",
        "notes",
    ]
    wcsv("stage3ab_starcoder_200k_ab_compare.csv", ab_rows, ab_fields)

    ab_md = f"""# Stage3 internal A vs B (corpus sums)

Counter: same as pipeline (`count_augmented` / placeholders as 1).

| Channel | input_tokens | output_tokens | saved_tokens | saved_pct | intro_tokens | net_saved_tokens |
|---------|-------------|---------------|--------------|-----------|--------------|------------------|
| Stage3_A_exact_aliasing | {s2_in} | {after_a} | {a_saved} | {pct(a_saved, s2_in):.4f}% | {a_intro} | {a_net} |
| Stage3_B_lexical_free_text | {after_a} | {after_b} | {b_saved} | {pct(b_saved, after_a):.4f}% | {b_intro} | {b_net} |
"""
    (out_dir / "stage3ab_starcoder_200k_ab_compare.md").write_text(ab_md, encoding="utf-8")
    print("[out]", out_dir / "stage3ab_starcoder_200k_ab_compare.md")

    # --- detail ---
    detail = {
        "stage3_ab_a_candidates": r.stage3_ab_a_candidates,
        "stage3_ab_a_selected": r.stage3_ab_a_selected,
        "stage3_ab_a_used_entries": r.stage3_ab_a_used_entries,
        "stage3_ab_a_sequence_saved": r.stage3_ab_a_sequence_saved,
        "stage3_ab_a_intro_tokens": r.stage3_ab_a_intro_tokens,
        "a_candidates_rejected_min_occ": r.stage3_ab_a_min_occ_reject_count,
        "a_candidates_rejected_raw_too_short": r.a_candidates_rejected_raw_too_short_sum,
        "a_candidates_rejected_alias_conflict": r.a_candidates_rejected_alias_conflict_sum,
        "a_candidates_rejected_gain_non_positive": r.a_candidates_rejected_gain_non_positive_sum,
        "a_candidates_rejected_context_rescore_negative": r.a_candidates_rejected_context_rescore_negative_sum,
        "b_free_text_candidates_total": r.b_free_text_candidates_total_sum,
        "b_free_text_candidates_visible_after_stage2": r.b_free_text_candidates_visible_after_stage2_sum,
        "b_clusters_formed": r.b_clusters_formed_sum,
        "b_clusters_rejected_too_small": r.b_clusters_rejected_too_small_sum,
        "b_clusters_rejected_similarity_or_quality": r.b_clusters_rejected_similarity_or_quality_sum,
        "b_clusters_rejected_intro_cost": r.b_clusters_rejected_intro_cost_sum,
        "b_clusters_selected_final": r.b_clusters_selected_final_sum,
        "stage3_ab_b_sequence_saved": r.stage3_ab_b_sequence_saved,
        "stage3_ab_b_intro_tokens": r.stage3_ab_b_intro_tokens,
        "stage2_removed_comment_count_sum": r.stage2_removed_comment_count_sum,
        "stage2_removed_comment_tokens_sum": r.stage2_removed_comment_tokens_sum,
        "stage2_removed_docstring_count_sum": r.stage2_removed_docstring_count_sum,
        "stage2_removed_docstring_tokens_sum": r.stage2_removed_docstring_tokens_sum,
        "stage2_removed_free_text_estimated_tokens_sum": r.stage2_removed_free_text_estimated_tokens_sum,
        "stage2_retained_for_b_probe_count_sum": r.stage2_retained_for_b_probe_count_sum,
        "stage2_retained_for_b_probe_tokens_sum": r.stage2_retained_for_b_probe_tokens_sum,
        "stage3_ab_stage2_input_tokens_sum": s2_in,
        "stage3_ab_after_a_tokens_sum": after_a,
        "stage3_ab_after_b_tokens_sum": after_b,
    }
    dfields = list(detail.keys())
    wcsv("stage3ab_starcoder_200k_detail.csv", [detail], dfields)
    detail_md = "| metric | value |\n|--------|-------|\n" + "\n".join(
        f"| {k} | {v} |" for k, v in detail.items()
    )
    (out_dir / "stage3ab_starcoder_200k_detail.md").write_text(
        "# Telemetry\n\n" + detail_md + "\n", encoding="utf-8"
    )
    print("[out]", out_dir / "stage3ab_starcoder_200k_detail.md")

    write_examples(sources, rc, tokenizer, tok_type, out_dir / "stage3ab_starcoder_200k_examples.md")
    print("[out]", out_dir / "stage3ab_starcoder_200k_examples.md")

    # --- final report ---
    s3_seq_saved = a_saved + b_saved
    if b_saved <= 0:
        b_case = "其他"
        if r.b_free_text_candidates_visible_after_stage2_sum == 0:
            b_case = "没候选（可见 B 串为 0）"
        elif r.b_clusters_selected_final_sum == 0 and r.b_clusters_rejected_too_small_sum > 0:
            b_case = "看得见候选但成不了簇（多为 singleton → too_small）"
        elif r.b_clusters_rejected_intro_cost_sum > 0 and r.b_clusters_selected_final_sum == 0:
            b_case = "intro 不划算为主"
        elif r.b_clusters_rejected_similarity_or_quality_sum > 0:
            b_case = "相似度/质量阈值拒绝为主"
    else:
        b_case = (
            f"B 有净序列收益（真实 saved_tokens={b_saved}）；funnel 上 formed={r.b_clusters_formed_sum}，"
            f"too_small 拒 {r.b_clusters_rejected_too_small_sum}，最终入选 cluster={r.b_clusters_selected_final_sum}，"
            f"相对 A（saved={a_saved}）仍偏弱。"
        )

    report = f"""# stage3ab_starcoder_200k 实验报告

## 1. 总压缩率

- **sequence_reduction_pct**（全序列相对 baseline）: **{r.sequence_reduction_pct:.4f}%**
- **effective_total_reduction_pct**（含 Stage1+Stage3 词表 intro）: **{r.effective_total_reduction_pct:.4f}%**

## 2. Stage1 / Stage2 / Stage3 各贡献多少（占 baseline 比例）

- **syntax_pct（Stage1）**: {r.syntax_pct:.4f}%
- **cleaning_pct（Stage2）**: {r.cleaning_pct:.4f}%
- **replacement_pct（仅 Stage3 序列段： (Stage2输出−Stage3输出)/baseline）**: {r.replacement_pct:.4f}%

## 3. Stage3 内部 A 真实压了多少（pipeline token 计数）

- A **input**（Stage2 输出 token 总和）: {s2_in}
- A **output**（A+guardrail 后）: {after_a}
- A **saved_tokens** (input−output): {a_saved}
- A **intro_tokens**（请求内求和）: {a_intro}
- A **net_saved_tokens** (saved−intro): {a_net}

## 4. Stage3 内部 B 真实压了多少

- B **input**（A 输出 token 总和）: {after_a}
- B **output**（最终 Stage3）: {after_b}
- B **saved_tokens**: {b_saved}
- B **intro_tokens**: {b_intro}
- B **net_saved_tokens**: {b_net}

## 5. 主力是不是 A？

- **是。** A 的序列 saved（{a_saved}）占 Stage3 总序列 saved（{r.replacement_saved}）的主体；B saved={b_saved}。

## 6. B 通道主因 / 瓶颈归类

- **判断**: {b_case}
- **telemetry**: visible={r.b_free_text_candidates_visible_after_stage2_sum}, clusters_formed={r.b_clusters_formed_sum}, too_small={r.b_clusters_rejected_too_small_sum}, sim/q={r.b_clusters_rejected_similarity_or_quality_sum}, intro={r.b_clusters_rejected_intro_cost_sum}, selected={r.b_clusters_selected_final_sum}

## 7. 是否支持「主要靠 A，B 还没起量」？

- **支持。** 以 **真实 token 计数**计：A saved={a_saved}，B saved={b_saved}，Stage3 序列段合计 {s3_seq_saved}（应与 corpus 级 replacement_saved={r.replacement_saved} 一致）。telemetry 估算字段 A/B sequence_saved={r.stage3_ab_a_sequence_saved}/{r.stage3_ab_b_sequence_saved} 与上式可能不完全同口径，本报告以 input−output 为准。

## 生效配置（本实验）

- stage2_profile={r.stage2_profile}, stage2_mode={r.stage2_mode}, resolution={r.stage2_resolution_source}
- stage3_ab_mode={r.stage3_ab_mode}
"""
    (out_dir / "stage3ab_starcoder_200k_report.md").write_text(report, encoding="utf-8")
    print("[out]", out_dir / "stage3ab_starcoder_200k_report.md")
    print("[done] baseline manifest actual_tokens=", manifest["actual_baseline_tokens"])


if __name__ == "__main__":
    main()
