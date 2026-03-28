# 实验与基准记录

本文件存放压缩率、示例数字及与其它基线的对照；**不**作为框架行为说明（见上级 `readme.md`）。

---

## 本地 Starcoder 约 1M tokens（`performance/code/data/starcoder_1m_tokens.txt`）

按约 **1M GPT-4 tokenizer tokens** 预算切分的 Python 样本，共 **301** 段。全链路 v2 评估脚本：

```bash
python performance/code/entropy_tokenizer_v2/eval/eval_local_starcoder_1m.py --tokenizers gpt4
python performance/code/entropy_tokenizer_v2/eval/eval_local_starcoder_1m.py --tokenizers gpt2
```

输出：`results/v2_compression_report.csv`、`results/v2_eval_detail.json`。只跑部分 tokenizer 时会合并进上述文件，不删除其它行。

语料按约 **1M GPT-4 tokens** 切分，故换用 GPT-2（更细粒度 BPE）时**同一段文本的 baseline token 数会明显变多**，降幅百分比不可与 GPT-4 行直接横向比绝对大小，仅作同 tokenizer 前后对照。

### GPT-4 tokenizer（tiktoken）

| 指标 | 数值 |
|------|------|
| 基线 tokens | 521,319 |
| 压缩后 tokens | 405,062 |
| 总降幅 | 22.30%（约 116,257 tokens） |
| Stage 1 / 2 / 3（占 baseline） | 0.93% / 8.42% / 12.95% |
| K* | 45 |
| Stage 3 替换词数 | 6,136 |

### GPT-2 tokenizer（HuggingFace `gpt2`）

| 指标 | 数值 |
|------|------|
| 基线 tokens | 1,002,719 |
| 压缩后 tokens | 515,466 |
| 总降幅 | 48.59%（约 487,253 tokens） |
| Stage 1 / 2 / 3（占 baseline） | 2.26% / 33.90% / 12.43% |
| V₀ | 50,257 |
| K* | 59 |
| Stage 3 替换词数 | 6,434 |

同一 CSV 中另有 `codegen-350M-mono`、`santacoder` 行（同源 301 段）。

---

## 玩具代码 Demo（`eval/run_v2.py demo`）

内置短脚本、GPT-4 tokenizer 的一次运行示例：

```
原始：2166 chars，469 tokens

  Stage 1：469 → 410   (−12.6% 相对本段 baseline)
  Stage 2：410 → 327
  Stage 3：327 → 314
  合计：469 → 314      (−33.0% 相对本段 baseline)
```

（具体 K*、替换词数随挖掘缓存而变，以终端输出为准。）

---

## Stage 1 与其它句法基线（可选对照）

在相同样本过滤规则下，仅比较**句法层**时，可运行：

```bash
python performance/code/entropy_tokenizer_v2/eval/eval_stage1_fair_compare.py
```

与仓库内 `Simpy-master` 评估脚本使用同一批 `starcoder_1m_tokens.txt` 子集时的过滤约定；细节以脚本内说明为准。

---

## MDL 接受算子示例（某次 demo / 玩具语料）

便于对照 `selected_skeletons` 字段含义（**非**固定基准）：

| 算子 | 骨架 | spi | freq | MDL 净收益 |
|------|------|-----|------|-----------|
| SYN_0 | `with {0}({1}, {2}, encoding={3}) as {4}:` | 8 | 2 | 14 |
| SYN_1 | `{0} = {1}.get({2}, {3})` | 5 | 2 | 8 |
| SYN_2 | `{0} = {1}.path.join({2}, f'...')` | 9 | 1 | 7 |
| SYN_3 | `for {0} in {1}.listdir({2}.input_dir):` | 9 | 1 | 7 |

spi = savings per instance。

---

## Demo 替换列表示例（玩具代码）

```
'DEFAULT_OUTPUT_DIR'  → <VAR>
'load_json_file'      → <VAR>
'"utf-8"'             → <STR>
'output_dir'          → <ATTR>
...
```
