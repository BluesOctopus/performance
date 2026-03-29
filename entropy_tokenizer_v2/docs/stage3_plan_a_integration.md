# Stage3 Plan A 接入主 Pipeline（literal codec）

## 目标

- 在 **固定 tokenizer** 下，将 `stage3/literal_codec` 的字段级 Plan A（variable / attribute / string）接到 `pipeline.apply_stage3` 与 `repo_miner` 挖掘链路。
- 与 **legacy** Stage3（`replacement_map` + `<VAR>` 等 placeholder）通过配置切换，**默认仍为 legacy**，避免破坏现有行为。

## 架构与接入点

| 组件 | 作用 |
|------|------|
| `config.py` | `ET_STAGE3_BACKEND`、`STAGE3_ESCAPE_PREFIX`、Plan A 类别与 `STAGE3_ARTIFACT_DIR` 等 |
| `repo_miner.py` | `mine_repo(..., stage3_backend=...)`：`plan_a` 时调用 `stage3/.../source_mining.mine_plan_a_from_sources`，写入 `RepoConfig.stage3_plan_a_*`；`legacy` 保持原 `replacement_map` |
| `repo_miner.load_plan_a_codebooks` | 将 JSON 中的码表反序列化为 `FieldCodebook`（带缓存） |
| `pipeline.apply_stage3` | `plan_a`：`encode_python_source_plan_a`；`legacy`：原 `apply_token_replacement_with_protected_spans` |
| `pipeline._stage3_vocab_intro` | `plan_a`：`build_plan_a_vocab_entries` + `compute_vocab_intro_cost`；`legacy`：原 placeholder 词条 |
| `placeholder_accounting.build_plan_a_vocab_entries` | Plan A 词条：`token = escape + V/A/S + code`，`definition = repr(literal)` |
| `stage3/literal_codec/pipeline/source_codec.py` | 源码级编解码：tokenize 边界、SYN 行保护、NAME/STRING 可逆格式 |
| `stage3/literal_codec/pipeline/source_mining.py` | 从语料收集字面量流、按 v2 同一 tokenizer 计成本并建码表 |
| `stage3/literal_codec/pipeline/v2_token_adapter.py` | 将 `marker_count.encode` 与 Plan A 的 `token_length` 对齐 |
| `eval/v2_eval.py` | `EvalResult` 扩展 `stage3_*` 字段；legacy 的 `n_replacement_words` 仍有意义；Plan A 时为 0 |
| `eval/run_v2.py` | `--stage3-backend`、`--stage3-output-tag` |

## 源码 ↔ 字段桥接

- **variable**：`tokenize.NAME` 且非 keyword/builtin 保护集；非 `attr` 上下文。
- **attribute**：`NAME` 且前一 token 为 `.`。
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

详见 `results/stage3_backend_comparison.csv`。结论要点：

- **gpt4 + Plan A**：仅看 **sequence-only** 时，压缩后 token 数可能 **高于** baseline（`__L__` 前缀在 tiktoken 下往往多 token），且 **corpus-once vocab intro** 条目数很大；这是真实测量结果，不是实现错误。
- **gpt2 + Plan A**：总压缩率仍为正，但 Stage3 的 `replacement_saved` 可能为负（字面量替换在序列上变长），需结合 `stage3_vocab_intro_tokens` 与业务是否摊销词表成本来解读。
- **legacy** 仍在中等规模 repo 上给出稳定的 Stage3 序列收益（placeholder 单 token 计数）。

## 测试

```bash
python -m pytest tests/ -q
cd stage3 && python -m pytest tests/ -q
```

## 已知限制与后续

1. **有效总 token**（序列 + vocab intro）尚未写入 `EvalResult.reduction_pct`；当前 reduction 仍为 sequence-only，与 legacy 口径一致。
2. Plan A 的 **mining 期望收益** 基于字面量级 tokenizer 成本；全文件重写后的交互（多 token 边界）可能不一致，需在报告中分开解读。
3. **f-string / number / 多行字符串** 未纳入 Plan A 压缩。
4. 方案 B/C 仍通过 `stage3/literal_codec/drift/` 等占位扩展。
