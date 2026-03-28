# 基于信息论的动态代码 Token 压缩框架

> **阶段性成果展示** — 最后更新：2026-02

---

## 一句话概括

我们正在开发一种**基于最小描述长度 (MDL) 原理的代码 token 压缩方案**：从代码语料中自动挖掘高频语法模式和词法模式，用"算子 token"替代它们，减少大模型处理代码时的 token 消耗。整个过程零硬编码、完全数据驱动，并用信息论框架来回答"应该选多少个算子"这个关键问题驱动**：
- 不预定义任何规则，从数据中**自动挖掘**压缩模式
- 用**信息论（MDL 原理）** 而非人工经验来决定最优压缩策略
- 框架通用，理论上可适配任意编程语言和代码库

---

## 核心方法

### 第一步：挖掘算子候选

从训练语料（StarCoderData ~100MB Python 代码）中挖掘两类模式：

**语法算子** — 来自 AST（抽象语法树）

把每条 Python 语句的变量名、常量都替换为占位符 `{0}`, `{1}`...，只保留"骨架"：

```
原始代码:  for i in range(10):        →  骨架:  for {0} in {1}({2}):
原始代码:  from os import path         →  骨架:  from {0} import {1}
原始代码:  if x == None:               →  骨架:  if {0} == {1}:
原始代码:  def calculate(a, b, c):     →  骨架:  def {0}({1}, {2}, {3}):
```

统计每种骨架在训练数据中出现多少次。出现 ≥ 50 次的进入候选池。共挖掘到 **1106 种 AST 模式**。

**词法算子** — 来自标识符命名习惯

```
前缀:  self.  np.  os.  get_  test_  add_  is_  set_  ...
后缀:  _name  _id  _data  _size  _path  _type  _list  ...
```

共挖掘到 **400 种词法模式**（200 前缀 + 200 后缀）。

### 第二步：用 MDL 信息论框架选择最优算子集合

这是我们工作的核心理论贡献。

#### 问题：加多少算子。

---

## 项目背景

### 为什么要做 token 压缩？

大模型（LLM）按 token 数量消耗计算资源。同一段代码，token 越少，推理越快、成本越低、能处理的上下文越长。

### 已有工作做了什么？

| 工作 | 思路 | 局限 |
|------|------|------|
| **SimPy** (AI Coders Are among Us) | 用 104 条人工规则重写 Python 代码：移除冒号/括号/逗号，压缩运算符，掩码字符串 | 规则是手工设计的，只适用于 Python，无法迁移到其他语言或数据 |
| **TokenSugar** | 提取高频词写入分词器 | 静态词表，不同代码库的高频词不同 |

### 我们想做什么不一样的？

**动态 + 理论合适？

直觉上，加越多算子 → 压缩越好。但每加一个算子，都有代价：

1. **词表变大了**：原来 100,000 个 token，现在变成 100,500 个。每个 token 需要更多 bit 来编码。
2. **要描述这个算子本身**：必须在"码本"中记录这个算子的模板是什么。

**MDL 原理** 把这个权衡形式化为一个优化问题：

```
L_total = L_data + L_codebook
        = (压缩后的 token 数) × log₂(扩展词表大小) + (码本 token 数) × log₂(原始词表大小)
```

- `L_data`：用扩展后的词表编码压缩后的数据需要多少 bit
- `L_codebook`：描述所有算子模板需要多少 bit

**目标：最小化 L_total。**

#### 贪心选择算法

把候选算子按收益排序后，逐个加入。每加一个，计算精确的边际变化：

```
ΔL = (N_after × log₂(V₀+k)) - (N_before × log₂(V₀+k-1)) + cb × log₂(V₀)
      └── 压缩后数据的编码代价 ──┘   └── 压缩前数据的编码代价 ──┘   └─ 码本代价 ─┘
```

当 ΔL ≥ 0 时停止 — 这就是 **MDL 最优点 K\***，代表"信息论意义上最划算的算子数量"。

### 第三步：评估

在 starcoderdata_100star 的 1000 个 Python 文件上，用三个主流分词器评估：
- **GPT-4** (tiktoken, 词表 100,277)
- **CodeGen-350M** (HuggingFace, 词表 50,257)
- **SantaCoder** (HuggingFace, 词表 49,152)

---

## 当前结果

### 压缩率（主要指标）

| 分词器 | Budget=100 | Budget=500 | Budget=1000 | 最佳 |
|--------|-----------|-----------|------------|------|
| GPT-4 | 4.0% | 5.9% | 6.2% | **6.3%** |
| CodeGen-350M | 3.9% | 6.3% | 7.0% | **7.2%** |
| SantaCoder | 4.1% | 6.6% | 7.4% | **7.5%** |

### MDL 信息论指标

| 分词器 | MDL 最优 K\* | K\*处压缩率 | L_total 最优 budget | 最优 MDL 降低 |
|--------|------------|-----------|-------------------|----|
| GPT-4 | **63** | 3.5% | ~500 | 5.5% |
| CodeGen-350M | **57** | 3.2% | ~1000 | 6.3% |
| SantaCoder | **54** | 3.4% | ~1000 | 6.6% |

**K\* 的含义**：MDL 贪心算法认为只需 ~60 个算子就到达了逐步添加的最优停止点。超过这个点，每个新算子的边际收益不足以抵消码本和词表膨胀的代价。

**L_total 最优 vs K\* 的差异**：全局 L_total 在 budget=500~1000 时最低。差异来自贪心逐步决策与全局评估的视角不同，这本身是可以在论文中分析的现象。

### MDL 拐点现象（以 GPT-4 为例）

| Budget | Token 压缩率 | MDL 压缩率 | 说明 |
|--------|------------|-----------|------|
| 200 | 4.9% | 4.8% | 两者同步上升 |
| 500 | 5.9% | **5.5%** | MDL 到达峰值 |
| 700 | 6.1% | 5.5% | token 还在涨，MDL 已饱和 |
| 1000 | 6.2% | **5.3%** | MDL 开始下降——过度压缩！ |
| 1500 | 6.3% | 5.2% | 继续恶化 |

这说明 **budget > 500 后继续加算子，从信息论角度看已经得不偿失**。这正是 MDL 框架的核心价值。

### 其他信息论度量

| 分词器 | Baseline 熵 (H) | Baseline bpb | 压缩后 bpb (B=500) |
|--------|---------------|------------|-------------------|
| GPT-4 | 10.98 bits | 4.03 | 3.80 |
| CodeGen-350M | 9.49 bits | 4.97 | 4.66 |
| SantaCoder | 9.68 bits | 4.36 | 4.08 |

- **H (熵)**：token 分布的信息密度。GPT-4 词表最大，熵最高。
- **bpb (bits-per-byte)**：跨分词器可比的归一化压缩效率指标。

### 与 SimPy 的对比

| 分词器 | 我们 (最佳) | SimPy (论文报告) | 差距 |
|--------|-----------|-----------------|------|
| GPT-4 | 6.3% | 10.4% | -4.1pp |
| CodeGen-350M | 7.2% | 13.5% | -6.3pp |
| SantaCoder | 7.5% | 8.8% | **-1.3pp** |

SantaCoder 上差距最小（7.5% vs 8.8%）。

---

## 差距分析：为什么不如 SimPy？

经过对 SimPy 源码的详细分析，差距来自**压缩范式的根本差异**：

### SimPy 做了但我们没做的事

| 类别 | SimPy 做法 | 我们的情况 | 预估影响 |
|------|-----------|-----------|---------|
| **标点移除** | 系统性删除所有 `:` `(` `)` `,` | 完全未处理 | **最大** |
| **运算符压缩** | `==`→`<eq>`, `+=`→`<augadd>` 等 24 种 | 完全未处理 | **大** |
| **字符串/注释掩码** | 内容替换为 `"MASK"` | 完全未处理 | **大** |
| **关键字 token 化** | `return`→`<return>` 等 31 种 | 仅通过 AST 骨架部分处理 | 中等 |
| **空白符规范化** | 移除多余空格 | 完全未处理 | 小 |

### 我们做了但 SimPy 没做的事

| 类别 | 说明 |
|------|------|
| **动态挖掘** | 零硬编码，1106 种 AST 模式全自动发现 |
| **词法前缀/后缀** | `self.`, `np.`, `get_`, `test_` 等 400 种 |
| **MDL 理论框架** | 自动最优 K\*，信息论度量体系 |
| **跨语言潜力** | 方法不依赖 Python 特定规则 |

### 核心结论

SimPy 是"**语法层面的全面改写**"（104 条手工规则覆盖几乎所有 Python 语法元素），我们是"**模式层面的选择性替换**"（自动发现高频重复模式）。两者并不矛盾，而是互补的。

---

## 项目结构

```
entropy_tokenizer/
├── config.py                    # 全局配置
├── run_pipeline.py              # 主流程入口 (data → mine → eval)
├── data_loader.py               # 从 StarCoderData 下载/缓存训练数据
├── frequency_miner.py           # 算子候选挖掘 (AST 骨架 + 词法前后缀)
├── compress_eval.py             # MDL 框架评估 (核心模块)
├── hierarchical_tokenizer.py    # [预留] 编解码外壳 (训练阶段)
├── embedding_init.py            # [预留] 新 token Embedding 初始化
├── lora_finetuning.py           # [预留] LoRA 微调脚本
├── requirements.txt
│
├── data/                        # 训练/测试数据 (~100MB Python 代码)
├── cache/                       # 挖掘结果 + 评估样本缓存
└── results/                     # 评估报告 (CSV + JSON)
    ├── compression_report_mdl.csv
    └── eval_detail_mdl.json
```

### 核心模块说明

| 模块 | 功能 |
|------|------|
| `frequency_miner.py` | 通用 AST 骨架挖掘（`copy.deepcopy` + `ast.NodeTransformer` + `ast.unparse()`）和词法前后缀挖掘 |
| `compress_eval.py` | MDL 目标函数定义、贪心最优选择 (`greedy_mdl_select`)、tokenizer-aware savings 计算、多分词器多 budget 评估、信息论指标（熵、bpb、MDL score）、报告生成 |
| `config.py` | 挖掘阈值、budget 列表、目标分词器、SimPy 对比基线 |

---

## 如何运行

### 环境准备

```bash
cd code/entropy_tokenizer
python -m pip install -r requirements.txt
```

### 运行完整流程

```bash
# 全流程（下载数据 + 挖掘 + 评估），约 15-20 分钟
python run_pipeline.py --all

# 快速测试（10MB 数据），约 3 分钟
python run_pipeline.py --all --quick

# 单独运行评估（需要先有挖掘结果）
python compress_eval.py
python compress_eval.py --tokenizers gpt4 santacoder --budgets 100 500 1000
```

### 输出

- `results/compression_report_mdl.csv`：所有分词器 × 所有 budget 的完整指标
- `results/eval_detail_mdl.json`：每个分词器的 Top-100 算子详情（含 MDL 边际收益）

---

## 已知不足

### 压缩率方面

1. **未处理标点/运算符/空白**：这是与 SimPy 差距的最大来源。代码中无处不在的 `:`, `,`, `()`, `==`, `+=` 等，SimPy 系统性地移除/压缩，而我们完全没有处理。
2. **AST 过度匿名化**：通用骨架挖掘将 `range`, `print`, `len` 等内置函数也替换为占位符，导致 `for {0} in range({1}):` 被泛化为 `for {0} in {1}({2}):`, 降低了每次匹配的节省量。
3. **单语句限制**：只挖掘单行语句级模式，无法捕捉多行代码块模式。
4. **理论 vs 实际 savings**：当前通过 token 算术估算压缩率，未实现真正的文本替换 + 重新编码验证。

### MDL 框架方面

5. **贪心 K\* 偏保守**：K\* ≈ 60 远低于经验最优 ≈ 500-1000。贪心逐步决策可能低估批量添加的协同效应。
6. **Uniform code 假设**：L_data 使用 `N × log₂(V)` 假设所有 token 等概率，实际上 token 分布高度不均匀。用实际熵替代 log₂(V) 可能给出更精确的 MDL 估计。

---

## 未来 TODO

### 短期（提升压缩率）

- [ ] **选择性匿名化**：在 AST 骨架中保留 Python 内置函数名（`range`, `print`, `len`, `isinstance` 等），只匿名化用户定义的标识符。预期 +1~2pp。
- [ ] **引入标点/运算符压缩规则**：将 SimPy 的标点移除和运算符压缩能力整合进框架，但用数据驱动的方式（统计频率后由 MDL 决定是否引入），而非硬编码。预期 +2~4pp。
- [ ] **算子重叠检测**：处理一条语句同时匹配多个算子的情况，避免 savings 高估。

### 中期（完善理论框架）

- [ ] **熵感知 MDL**：用实际 token 熵替代 uniform code 的 log₂(V) 假设
- [ ] **真实压缩验证**：实现实际的文本替换 + 重新编码，验证理论估算的准确性
- [ ] **多语言扩展**：在 Java/JavaScript 等语言上验证框架通用性

### 长期（模型训练验证）

- [ ] 使用 `hierarchical_tokenizer.py` 实现真正的编解码
- [ ] 使用 `embedding_init.py` 进行新 token Embedding 均值池化初始化
- [ ] 使用 `lora_finetuning.py` 进行参数高效微调
- [ ] 在代码生成/补全下游任务上评估端到端效果

---

## 依赖

```
transformers
datasets
tiktoken
tqdm
huggingface_hub
torch
peft          # 仅训练阶段需要
```

---

## 参考工作

1. **SimPy** — *AI Coders Are among Us: Rethinking Programming Language Grammar towards Efficient Code Generation*
2. **TokenSugar** — *Token Sugar: Making Source Code Sweeter for LLMs through Token-Efficient Shorthand*
3. **DietCode** — *DIETCODE: Automatic Optimization for Dynamic Tensor Programs*
4. **SlimCode** — *Natural Is the Best: Model-Agnostic Code Simplification for Pre-trained Large Language Models*
