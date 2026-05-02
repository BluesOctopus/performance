# entropy_tokenizer_v2

面向 Python 源码的 **按仓库动态挖掘** 的 token 压缩管线：在信息论式启发下，依次经过 **句式算子替换 → 文本清洗 → 高分词类别占位**，输出压缩文本与分阶段 token 计数（增广词表模拟见 `marker_count.py`）。

**特点概览**

- **动态**：每个语料 / 仓库单独挖掘 `RepoConfig`（句式算子 + 替换词表），缓存在 `cache/`。
- **三阶段串联**：Stage 1 句法、Stage 2 清洗、Stage 3 词级替换；各阶段职责与实现文件见下文。
- **规则可配**：清洗开关、MDL 与打分超参在 `config.py` 中集中配置。

实验数字与跑分见 **`docs/EVAL_RESULTS.md`**；机器可读汇总见 **`results/`**（由评估脚本生成）。

---

## 三阶段总览

```
原始 Python 代码
       │
       ▼  Stage 1 —— 句式压缩（syntax_compressor.py）
       │   AST 遍历 → 句式骨架统计 → 候选池（语料级经验省 token）→ MDL 贪心选 K* 条骨架
       │   将命中语句的 header 行替换为 `<SYN_N>` + 槽值（空格分隔）
       │
       ▼  Stage 2 —— 清洗（lossy_cleaner.py；eval 里对 SYN 行单独策略，见下）
       │   删空行、行尾空白；可选去掉缩进；评估管线中可保留注释/docstring（见 Stage 2 小节）
       │
       ▼  Stage 3 —— Token 打分与替换（token_scorer.py）
       │   `tokenize` 抽词频 → Score(w) → 分位数截断 → 建 replacement_map → 按行替换
       │
       ▼
  压缩后文本
```

**挖掘与压缩的次序**（`repo_miner.mine_repo`）：先对语料做 **仅无损** 清洗（`lossless_clean`：**保留** `#` 注释与 docstring；删空行、行尾空白，**保留**缩进），再统计 baseline tokens、挖骨架、算 Stage 3 词表；**压缩一条源码**时则按 `eval/v2_eval.py` 中 `apply_v2_compression` 的顺序执行 Stage 1 → 2 → 3（Stage 2 的配置与挖掘阶段不完全相同，见 Stage 2）。

---

## Stage 1：句式骨架与 MDL

**作用**：把反复出现的 AST「句法模式」收成少量算子 token，把一整段语句头（含关键字）压成一行 `` `<SYN_N>` `` 加槽值。

**骨架怎么来**  
对语句节点计算匿名化模板字符串（槽位 `{0},{1},…`），在语料上统计每种骨架出现次数；仅频率不低于 `AST_MIN_FREQ`（`config.py`）的进入候选池。

**候选如何排序**  
`build_candidate_pool` 在真实语料上按与压缩一致的方式，估算「每类骨架替换后」能省多少 token（`marker_count` 口径下的计数），得到 `empirical_total_savings`；再与码本开销 `MDL_CODEBOOK_OVERHEAD` 组合后排序，供 `greedy_mdl_select` 使用。

**MDL 贪心接受**  
在语料总 token 数 `N_baseline`、基础词表大小 `V₀` 下，逐个尝试加入骨架；若加入后总描述长度下降则接受（实现见 `syntax_compressor.greedy_mdl_select`）。判据形式为：

```text
ΔL_k = N_new · log₂(V₀+k) − N_curr · log₂(V₀+k−1) + cb_k · log₂(V₀) < 0
```

- `N_curr` / `N_new`：接受前后语料总 token 数（与 `empirical_total_savings` 衔接）。
- `cb_k`：`MDL_CODEBOOK_OVERHEAD`。
- `V₀`：目标 tokenizer 词表规模。

**压缩时**  
对源码 AST 匹配已选骨架，将对应 **header 行** 替换为一行 `` `<SYN_N> slot0 slot1 …``（具体格式见 `syntax_compressor.compress_source_syntax`）。

**持久化**  
`cache/repo_config_<tok>_<name>.json` 的 `selected_skeletons` 第 `N` 条对应 `` `<SYN_N>` ``。

---

## Stage 2：有损 / 无损清洗

**实现文件**：`lossy_cleaner.py`。单文件入口为 `clean_code(source, CleaningConfig)`；规则按固定顺序执行：

| 代号 | 规则 | 默认是否有损 | 说明 |
|------|------|--------------|------|
| R05 | 删除模块/类/函数首行 docstring | 有损 | 依赖 AST 定位；失败时回退正则 |
| R01 | 删除 `#` 行注释 | 无损 | `tokenize` 实现，避免误伤字符串内 `#` |
| R03 | 行尾空白 | 无损 | |
| R02 | 删除空行 | 无损 | |
| R04 | 去掉行首缩进（每行 `lstrip`） | 有损 | 结构破坏，压缩后通常不可再 parse |

**配置表**（`config.CLEANING_RULES`）与 `CleaningConfig` 字段一一对应，可开关各条。

**挖掘阶段**（`repo_miner`）  
只对语料调用 **`lossless_clean`**：等价于开启 R02、R03，**关闭** R01、R04、R05（**不删**注释与 docstring）。目的是规整空白、便于稳定解析与统计。

**全链路评估阶段**（`eval/v2_eval.py` → `_clean_stage2_skip_syn`）  
策略与「整文件 `clean_code`」不同，要点如下：

1. **Stage 1 产生的行**（匹配 `` `^\s*<SYN_\d+>` ``）：整行 **冻结**，只做 `rstrip`，不删注释、不剥缩进，避免破坏算子行格式。
2. **其余行**：按行调用 `clean_code`，且配置为 **删空行、删行尾空白、去掉缩进**，但 **不删** 行注释与 docstring（与挖掘用的 `lossless_clean` 不同，便于在评估里保留部分语义信息）。

因此：**Stage 2 在 README 里单独成章**；若你直接调用 `lossy_clean` / `lossy_cleaner` 的默认 `CleaningConfig()`，行为又与 `v2_eval` 里 Stage 2 不一致，以实际调用的入口为准。

---

## Stage 3：Score、类别占位与替换

**词表从哪来**  
对源码用 `tokenize` 扫描，按类别累计频次：标识符、`.` 右侧属性名、字符串、f-string、数字（见 `token_scorer._extract_vocab_from_source`）。

**Score**（与 `token_scorer.compute_scores` 一致）：

```text
Score(w) = ΔT(w) / (ΔI(w) + ε)
```

- `p(w) = freq(w) / Σ_u freq(u)`，跨类别统一分母。
- `spt(w)`：用目标 tokenizer 对 `w` 单独 `encode` 的长度（若不可用则字符长启发式）。
- `ΔT(w) = max(0, spt(w) − 1) × freq(w)`；仅 `spt > 1` 的词在替换候选里有意义。

| 符号 | 含义 | 计算方式 |
|:----:|------|----------|
| ΔT(w) | 整块换成 1 个类别占位符时，相对 subtoken 计数节省 × 频次 | (spt − 1) × freq |
| ΔI(w) | 自信息 | −log₂ p(w) |
| ε | 平滑 | `SCORE_EPSILON`（`config.py`） |

**选谁替换**  
`select_replacement_set`：在 `spt > 1` 的词里，按 Score 排序，取分位数以上（`SCORE_THRESHOLD_PERCENTILE`，默认约前 30% 高分）。`build_replacement_map` 把类别映射到 `PLACEHOLDERS`（`` `<VAR>` ``、`` `<ATTR>` ``、`` `<STR>` ``、`` `<FSTR>` ``、`` `<NUM>` ``）。

**保护词**  
关键字、内置名、`self` / `cls` / `args` / `kwargs` 等不参与替换（`token_scorer._PROTECTED`）。

**怎么写回文本**  
`apply_token_replacement`：先整段匹配替换字符串/数字字面量，再替换标识符；**在 `v2_eval` 里对含 `` `<SYN_N>` `` 的行整行跳过**，只在非 SYN 行上做替换，避免与 Stage 1 冲突。

---

## 增广词表下的 token 计数

`` `<SYN_N>` ``、`` `<VAR>` `` 等在真实 tokenizer 中会被拆成多个 subtoken。评估统一用 **`marker_count.count_augmented()`**：先去掉占位再 encode，再按占位出现次数加回（每个占位计 1）。

---

## Per-Repo 动态性

同一批源码在不同仓库上挖掘，会得到不同的骨架集合与替换集；入口见 `repo_miner.mine_from_sources` / `mine_from_repo_path`，配置序列化为 JSON 后供 `eval` 与压缩 API 使用。

---

## 目录结构（精简后）

```
entropy_tokenizer_v2/
├── config.py
├── lossy_cleaner.py
├── markers.py
├── marker_count.py
├── pipeline.py
├── repo_miner.py
├── syntax_compressor.py
├── token_scorer.py
├── stage2/
│   ├── __init__.py
│   ├── cleaning.py
│   └── config.py
├── cache/
├── results/
├── docs/
│   └── EVAL_RESULTS.md
└── eval/
    ├── bootstrap_v2.py
    ├── run_v2.py
    └── v2_eval.py
```

### 模块职责

- `pipeline.py`：核心压缩业务实现（Stage1/2/3 串联、skip-SYN 处理、单文件分解结果）。
- `eval/v2_eval.py`：评估入口（采样、聚合、统计、报表与结果落盘），不承载底层阶段实现。
- `markers.py` / `marker_count.py`：marker 识别与增广 token 计数的统一底层。

---

## 使用方法

路径以仓库根目录（`paper`）为准：

```bash
python performance/code/entropy_tokenizer_v2/eval/run_v2.py demo --tokenizer gpt4
python performance/code/entropy_tokenizer_v2/eval/run_v2.py demo --file path/to/script.py --tokenizer gpt4
python performance/code/entropy_tokenizer_v2/eval/run_v2.py eval --repo /path/to/project --tokenizers gpt4
python performance/code/entropy_tokenizer_v2/eval/run_v2.py eval --samples 200 --tokenizers gpt4 santacoder
```

---

## 设计说明（非实验结论）

| 点 | 当前实现 |
|----|----------|
| 有损清洗 | R04/R05 会破坏可解析性；若需可逆压缩，应在配置中关闭 |
| Stage 2 两套策略 | 挖掘用 `lossless_clean`；`v2_eval` 全链路用「按行 + 冻结 SYN」逻辑，见 Stage 2 小节 |
| Stage 3 替换范围 | 含槽值内 identifier；若需保护可在挖掘阶段收窄词表 |
| 动态性 | 按仓库 / 语料子集挖掘一次，缓存复用 |
