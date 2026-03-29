# Stage3 Plan A 接入主 Pipeline（literal codec）

## 目标

- 在 **固定 tokenizer** 下，将 `stage3/literal_codec` 的字段级 Plan A（variable / attribute / string）接到 `pipeline.apply_stage3` 与 `repo_miner` 挖掘链路。
- 与 **legacy** Stage3（`replacement_map` + `<VAR>` 等 placeholder）通过配置切换，**默认仍为 legacy**，避免破坏现有行为。

## 阻断性修复（成本与统计口径）

以下问题曾在早期实现中导致 **expected_gain 与实测 replacement_saved 严重背离**（含 gpt4 下负压缩假象），已通过代码修复；**不是**普通超参调优：

1. **variable / attribute 互斥**：`NAME` 在 `.` 后只计入 **attribute**，否则只计入 **variable**，禁止同一标识符双计数。
2. **真实表面成本**：建码与 `expected_gain` 使用 `surface_cost.encoded_form_token_cost`（与 `source_codec` 写回形态一致：`__L__V/A{code}`、字符串 `repr(f"__L__S{code}")`），**不用**裸 `code` 长度。
3. **vocab intro 仅对已用码计费**：`extract_used_plan_a_entries` + `build_used_plan_a_vocab_entries`，与 legacy「仅实际出现的 placeholder」思路对齐；**不对**整本码表一次性摊销。
4. **sequence 评测**：Plan A 仍按 **真实 tokenizer 长度** 计 sequence token；**不**把 `__L__...` 当作单 token placeholder。

`pipeline._stage3_vocab_intro` 在 `plan_a` 下对 **used entries** 构建词条并计费（见 `placeholder_accounting.build_used_plan_a_vocab_entries`）。

## 架构与接入点

| 组件 | 作用 |
|------|------|
| `config.py` | `ET_STAGE3_BACKEND`、`STAGE3_ESCAPE_PREFIX`、Plan A 类别与 `STAGE3_ARTIFACT_DIR` 等 |
| `repo_miner.py` | `mine_repo(..., stage3_backend=...)`：`plan_a` 时调用 `stage3/.../source_mining.mine_plan_a_from_sources`，写入 `RepoConfig.stage3_plan_a_*`；`legacy` 保持原 `replacement_map` |
| `repo_miner.load_plan_a_codebooks` | 将 JSON 中的码表反序列化为 `FieldCodebook`（带缓存） |
| `pipeline.apply_stage3` | `plan_a`：`encode_python_source_plan_a`；`legacy`：原 `apply_token_replacement_with_protected_spans` |
| `pipeline._stage3_vocab_intro` | `plan_a`：**仅 used** `build_used_plan_a_vocab_entries` + `compute_vocab_intro_cost`；`legacy`：原 placeholder 词条 |
| `placeholder_accounting.build_used_plan_a_vocab_entries` | 仅实际出现在压缩源码中的 `(field, code)` 词条 |
| `stage3/literal_codec/pipeline/source_codec.py` | 源码级编解码：tokenize 边界、SYN 行保护、NAME/STRING 可逆格式 |
| `stage3/literal_codec/pipeline/source_mining.py` | 从语料收集字面量流、按 v2 同一 tokenizer 计成本并建码表 |
| `stage3/literal_codec/pipeline/v2_token_adapter.py` | 将 `marker_count.encode` 与 Plan A 的 `token_length` 对齐 |
| `eval/v2_eval.py` | `EvalResult` 扩展 `stage3_*` 字段；legacy 的 `n_replacement_words` 仍有意义；Plan A 时为 0 |
| `eval/run_v2.py` | `--stage3-backend`、`--stage3-output-tag` |

## 源码 ↔ 字段桥接

- **attribute**：`NAME` 且前一 token 为 `.`（**仅**计入 attribute，**不**再计入 variable）。
- **variable**：`tokenize.NAME` 且非 keyword/builtin 保护集，且**非** attribute 上下文。
- **string**：非 f-string、无换行的 `STRING`；`ast.literal_eval` 成功且 inner 不以 `escape_prefix` 开头（避免与协议混淆）。
- **number / f-string**：当前 **不压缩**（保守跳过）。
- **压缩 NAME 形态**：`{escape_prefix}{V|A}{code}`，整体为合法标识符。
- **压缩 STRING 形态**：`repr(f"{escape_prefix}S{code}")`，解码时查表恢复 **原始引号形式** 的字面量。

## 可逆性

- 在参与替换的 token 上：`decode(encode(source)) == source`（测试见 `tests/test_stage3_plan_a_source_roundtrip.py`）。
- `get_syn_line_spans` 保护的 SYN 行不替换。

## 运行命令（可复现）

在项目根目录 `entropy_tokenizer_v2/`：

```bash
# Legacy + gpt4
python eval/run_v2.py eval --repo . --samples 80 --tokenizers gpt4 --stage2-profile stage2_aggressive --stage3-backend legacy --stage3-output-tag _s3legacy_gpt4

# Plan A + gpt4（生成 v2_compression_report_stage3_plan_a.csv）
python eval/run_v2.py eval --repo . --samples 80 --tokenizers gpt4 --stage2-profile stage2_aggressive --stage3-backend plan_a --stage3-output-tag _stage3_plan_a

# gpt2 对照
python eval/run_v2.py eval --repo . --samples 80 --tokenizers gpt2 --stage2-profile stage2_aggressive --stage3-backend legacy --stage3-output-tag _s3legacy_gpt2
python eval/run_v2.py eval --repo . --samples 80 --tokenizers gpt2 --stage2-profile stage2_aggressive --stage3-backend plan_a --stage3-output-tag _gpt2_plan_a
```

环境变量（可选）：

- `ET_STAGE3_BACKEND=legacy|plan_a`
- `ET_STAGE3_ESCAPE_PREFIX`（默认 `__L__`）
- `ET_STAGE3_PLAN_A_ENABLED_CATEGORIES`（默认 `variable,attribute,string`）
- `ET_STAGE3_PLAN_A_MIN_GAIN`
- `ET_STAGE3_ARTIFACT_DIR`

## 产物路径

| 产物 | 路径 |
|------|------|
| Plan A 码表/报告（挖掘时写入） | `results/stage3_plan_a/codebook_<tok>_sources.json`、`report_*.json` |
| 带语料标签的副本 | `results/stage3_plan_a/codebook_gpt4_entropy_tokenizer_v2.json` 等 |
| 单次评测 CSV/JSON | `results/v2_compression_report*.csv`、`results/v2_eval_detail*.json` |
| 正式 Plan A 报告名 | `results/v2_compression_report_stage3_plan_a.csv`、`results/v2_eval_detail_stage3_plan_a.json` |
| 对照表 | `results/stage3_backend_comparison.csv` |
| 挖掘缓存 | `cache/repo_config_<tok>_<name>_<backend>.json` |

## 实验结果摘要（本地 repo，80 文件）

以 `results/stage3_backend_comparison.csv` 为准（与 `v2_compression_report_*.csv` 同步刷新；表内含 `sequence_*` 与 `effective_total_*`）。当前评测在 **互斥采集 + real_surface_form + used-only vocab** 下重跑。

- **主指标**：`effective_total_reduction_pct`（整语料共享 Stage1/3 词表后的最终压缩率）；**对照**：`sequence_reduction_pct`（仅正文序列）。
- **当前矩阵（gpt4/gpt2 × legacy/plan_a）**：在同一 80-file 切片上，**legacy 的 effective-total 压缩率均高于 plan_a**；plan_a 常见 **Eff% < Seq%**，因 Stage3 used-only 词条 intro 在 tokenizer 下仍显著。
- **legacy vs Plan A 横向比较**：以各自行的 `effective_total_*` 为准；两种 backend 的序列计数规则不同，勿只比 raw `final_tokens` 绝对值。

## 测试

```bash
python -m pytest tests/ -q
cd stage3 && python -m pytest tests/ -q
```

## 评测指标：序列 vs 整语料有效总成本

`eval/v2_eval.py` 的 `EvalResult` 同时给出：

| 字段 | 含义 |
|------|------|
| `sequence_final_tokens` / `sequence_reduction_pct` | 仅 **正文序列** token（占位符计 1）；与历史 `final_tokens` / `reduction_pct` 同义。 |
| `effective_total_tokens` / `effective_total_reduction_pct` | **整语料共享词表后的最终总成本**：`sequence_final_tokens + corpus_once_vocab_intro`。 |
| `final_vocab_intro_tokens` | `stage1_vocab_intro_tokens + stage2_vocab_intro_tokens + stage3_vocab_intro_tokens`（当前 Stage2 无独立 intro，为 0）。 |

**Corpus-once**：Stage1 骨架词条只计一次；Stage3 legacy 为全语料出现过的 placeholder 并集；Stage3 Plan A 为 **used-only** 的 `(field, code)` 词条。评测循环中 **不按文件重复叠加** intro。

解读实验结论时，**优先看 `effective_total_reduction_pct`**；`sequence_reduction_pct` 保留用于与旧结果、纯序列对比对齐。

## 已知限制与后续

1. Plan A 的 **mining 期望收益** 已与写回源码的 **surface form**（`__L__V/A/S`、字符串 `repr(...)`）对齐；全文件级仍可能存在上下文与边界效应，但与「裸 code 长度」类旧 bug 无关。
2. **f-string / number / 多行字符串** 未纳入 Plan A 压缩。
3. 方案 B/C 仍通过 `stage3/literal_codec/drift/` 等占位扩展。
