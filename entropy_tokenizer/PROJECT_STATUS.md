# 项目状态总结：基于算子的层次化代码分词压缩

> 最后更新：2025-01

---

## 1. 项目目标

通过在现有 BPE 分词器之上引入**动态算子 (Operator) 层**，减少代码文本的 token 消耗量，从而提升大模型在代码任务上的性能。

**对标工作**：SimPy（AST 结构简写）、TokenSugar（高频词缩写）

**核心差异化**：SimPy/TokenSugar 的模式是手动预定义的，本方案从数据中**动态挖掘**算子，零硬编码，可适配任意代码语料。

---

## 2. 项目结构

```
entropy_tokenizer/
├── config.py                    # 全局配置：数据路径、挖掘阈值、评估参数
├── run_pipeline.py              # 主流程入口：data → mine → eval
├── data_loader.py               # Step 0: 从 StarCoderData 流式下载并缓存训练/测试数据
├── frequency_miner.py           # Step 1: 算子候选挖掘（AST 语法 + 词法前后缀）
├── compress_eval.py             # Step 2: 多分词器压缩率评估
├── hierarchical_tokenizer.py    # [预留] 基于 regex 的编解码外壳（未来训练阶段使用）
├── embedding_init.py            # [预留] 新算子 token 的 Embedding 均值池化初始化
├── lora_finetuning.py           # [预留] LoRA 微调训练脚本
├── requirements.txt             # 依赖列表
├── dynamic_operator_architecture.md  # 早期架构设计文档
│
├── data/                        # 缓存的训练/测试数据集（Arrow 格式）
│   ├── train/                   #   训练集（~100MB，19024 个 Python 文件）
│   └── test/                    #   测试集
│
├── cache/                       # 中间结果缓存
│   ├── mining_results.json      #   挖掘结果：AST 模式 + 词法前后缀 + 频次
│   └── eval_100star_samples.json#   评估用 starcoderdata_100star 前 1000 个样本
│
└── results/                     # 评估输出
    ├── compression_report.csv   #   压缩率报告（所有分词器 × 所有 budget）
    └── eval_detail.json         #   详细结果：每个分词器的 Top-50 算子列表
```

### 文件功能详解

| 文件 | 状态 | 说明 |
|------|------|------|
| `config.py` | **活跃** | 集中管理所有超参数：`AST_MIN_FREQ=50`, `LEXICAL_MIN_FREQ=100`, `OPERATOR_BUDGETS=[20,50,100,200,500,1000]`，三个目标分词器（GPT-4, codegen-350M, santacoder），SimPy 对比基线 |
| `run_pipeline.py` | **活跃** | 三步流水线：`--step data`（下载）→ `--step mine`（挖掘）→ `--step eval`（评估），或 `--all` 全流程 |
| `data_loader.py` | **活跃** | 从 HuggingFace `bigcode/starcoderdata` 流式下载 Python 代码，按 100MB 上限采样，80/20 划分 train/test |
| `frequency_miner.py` | **活跃** | **核心模块** — 通用 AST 骨架挖掘（`copy.deepcopy` + `ast.unparse` + 占位符替换），词法前后缀挖掘（regex） |
| `compress_eval.py` | **活跃** | **核心模块** — tokenizer-aware savings 计算、候选池排序、多 budget 消融评估、对比报告生成 |
| `hierarchical_tokenizer.py` | **预留** | regex 编解码外壳，供未来模型训练阶段使用（将算子 token 注入实际文本流） |
| `embedding_init.py` | **预留** | 新 token 的 Embedding 初始化（均值池化策略） |
| `lora_finetuning.py` | **预留** | LoRA 参数高效微调脚本，绑定 `hierarchical_tokenizer` + `embedding_init` |

---

## 3. 技术方案

### 3.1 两类算子

**语法算子 (Syntax Operators)** — 来自 AST 结构骨架

- 对每条 Python 语句级 AST 节点，将所有变量名 (`ast.Name`)、字面量 (`ast.Constant`)、函数/类名等替换为 `{0}`, `{1}`, ... 占位符
- 剥离函数体/循环体等 block body，只保留**语句头**
- 使用 `ast.unparse()` 重建骨架字符串
- 示例：`for i in range(10):` → `for {0} in {1}({2}):`

**词法算子 (Lexical Operators)** — 来自标识符命名模式

- 前缀：`self.`, `np.`, `get_`, `test_`, `os.`, `add_` 等
- 后缀：`_name`, `_id`, `_data`, `_size`, `_path` 等
- 通过 regex 在代码文本中统计频次

### 3.2 算子选择策略

对每个候选算子，计算 **tokenizer-aware savings**：

```
savings_per_instance (spi) = tokens(fixed_part) - 1
total_savings_est = spi × frequency_in_training_data
```

其中 `fixed_part` 是骨架中去除占位符后的固定文本。按 `total_savings_est` 降序排列，取 Top-K 个算子（K = budget）。

### 3.3 压缩率评估

在 `starcoderdata_100star` 的前 1000 个样本上：
1. 计算 baseline token 数
2. 对每个文件重新解析 AST、匹配选中的语法算子骨架，计数匹配次数
3. 对每个文件用 regex 匹配选中的词法算子，计数匹配次数
4. `compressed = baseline - Σ(spi × match_count)`
5. `reduction% = (1 - compressed/baseline) × 100`

---

## 4. 实验迭代历程

### V1：硬编码 AST 模式（已弃用）

手动为每种 AST 节点类型编写 `isinstance` 判断和骨架模板：

```python
if isinstance(node, ast.For):
    return f"for {{0}} in range({{1}}):"  # 手动保留 range
```

- 优点：高 spi（保留了 `range`, `isinstance`, `print` 等函数名作为固定文本）
- 缺点：硬编码约 105 种模式，覆盖有限，方法论贡献弱

**V1 最佳结果**（budget=500）：

| Tokenizer | Reduction |
|-----------|-----------|
| gpt4 | 6.9% |
| santacoder | 8.1% |

### V2：通用 AST 挖掘（当前版本）

用 `copy.deepcopy` + `ast.NodeTransformer` + `ast.unparse()` 自动生成骨架，**零硬编码**：

- 所有 `ast.Name` 节点 → `{N}` 占位符
- 所有非布尔 `ast.Constant` → `{N}` 占位符
- 函数名、类名、参数名、模块名等字符串属性 → `{N}` 占位符
- block body → `pass`（只保留头部）

**V2 结果**：

| Tokenizer | B=20 | B=50 | B=100 | B=200 | B=500 | B=1000 |
|-----------|------|------|-------|-------|-------|--------|
| gpt4 | 2.5% | 3.3% | 4.0% | 4.9% | 5.9% | **6.2%** |
| codegen-350M-mono | 2.3% | 3.1% | 3.9% | 4.8% | 6.3% | **7.0%** |
| santacoder | 2.3% | 3.3% | 4.1% | 5.1% | 6.6% | **7.4%** |

**挖掘统计**：
- AST 模式：1106 个（min_freq ≥ 50），Top-20 示例：
  - `{0} = {1}`, `{0}`, `def {0}({1}):`, `from {0} import {1}`, `import {0}`, `return {0}`, `def {0}({1}, {2}):`, `{0}({1})`, `{0} = {1}({2})`, `if {0} == {1}:`
- 词法前缀：200 个，Top-5：`self.`, `np.`, `get_`, `test_`, `os.`
- 词法后缀：200 个，Top-5：`_name`, `_id`, `_data`, `_size`, `_path`

---

## 5. 与 SimPy 的对比

| Tokenizer | 本方案 V2 (B=1000) | SimPy (论文报告) | 差距 |
|-----------|---------------------|------------------|------|
| gpt4 | 6.2% | 10.4% | -4.2pp |
| codegen-350M-mono | 7.0% | 13.5% | -6.5pp |
| santacoder | 7.4% | 8.8% | -1.4pp |

### 差距分析

1. **过度匿名化**：通用方案将所有 `Name` 节点都匿名化，包括 `range`, `print`, `isinstance` 等高频函数名。这导致骨架如 `for {0} in range({1}):` 被泛化为 `for {0} in {1}({2}):`，spi 降低（`range` 不再是固定文本的一部分）。

2. **评估方式差异**：本方案评估的是理论压缩上界（不考虑算子间的重叠），SimPy 可能使用了不同的实验设置和数据集。

3. **未利用的优势**：
   - 通用方案发现了 1106 个 AST 模式（远多于 SimPy 的手工定义），budget 从 500→1000 仍有增长，说明还有提升空间
   - 词法算子贡献显著（占总 savings 的 60-75%），但目前只挖掘了前后缀两种模式

---

## 6. 已知问题与改进方向

### 6.1 可立即执行的改进

| 改进项 | 预期效果 | 说明 |
|--------|----------|------|
| **选择性匿名化** | +1~2pp | 保留函数调用位置的 Name 节点（如 `range`, `print`, `isinstance`），仅匿名化变量和常量，恢复高 spi 的具体模式 |
| **增加训练数据** | +0.5~1pp | 当前 100MB 训练集仅 19K 文件，增大至 500MB 可发现更多低频但高 spi 的模式 |
| **词法模式扩展** | +0.3~0.5pp | 增加 camelCase 前缀、双下划线命名、装饰器名等挖掘类型 |

### 6.2 中期改进

- **算子重叠检测**：当前未处理一条语句同时匹配多个算子的情况（可能导致 savings 被高估）
- **Try/ExceptHandler 联合建模**：当前 `ast.Try` 节点被跳过，仅单独处理 `ExceptHandler`
- **压缩率的真实验证**：当前是理论估算（token 算术），应实现真正的文本替换 + 重新编码来验证

### 6.3 远期（训练阶段）

- 使用 `hierarchical_tokenizer.py` 实现真正的编解码
- 使用 `embedding_init.py` 初始化新 token 的 Embedding
- 使用 `lora_finetuning.py` 进行参数高效微调
- 在下游任务（代码生成、代码补全）上评估端到端效果

---

## 7. 运行指南

### 环境准备

```bash
cd code/entropy_tokenizer
python -m pip install -r requirements.txt
```

### 完整流程

```bash
# 全流程（下载数据 + 挖掘 + 评估）
python run_pipeline.py --all

# 快速测试（10MB 数据子集）
python run_pipeline.py --all --quick

# 单独运行各步骤
python run_pipeline.py --step data          # 下载数据
python run_pipeline.py --step mine          # 运行挖掘
python run_pipeline.py --step eval          # 运行评估

# 自定义评估
python run_pipeline.py --step eval --tokenizers gpt4 santacoder --budgets 100 500 1000
```

### 运行耗时参考

| 步骤 | 耗时 |
|------|------|
| `data`（首次下载 100MB） | ~5 min |
| `mine`（19K 文件 AST + 词法挖掘） | ~3.5 min |
| `eval`（3 分词器 × 6 budget × 1000 样本） | ~4 min |

---

## 8. 依赖

```
transformers
datasets
tiktoken
tqdm
huggingface_hub
torch
peft          # 仅训练阶段需要
```
