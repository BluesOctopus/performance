# SimPy & TokenSugar 实验设置整理

## SimPy (AI Coders Are Among Us)

### 训练数据集

| 数据集 | 来源 | 说明 |
|:---|:---|:---|
| starcoderdata_100star_py | `zhensuuu/starcoderdata_100star_py` (HuggingFace) | StarCoderData 的 Python 子集，筛选 100 star 以上仓库 |

### Token 压缩率测试模型 / Tokenizer

SimPy 在 15 个模型 tokenizer 上测试压缩率，数据为 `starcoderdata_100star` 前 1000 个样本：

| 模型 / Tokenizer | 类型 |
|:---|:---|
| Codex (code-davinci-002) | OpenAI 闭源 |
| GPT-4 | OpenAI 闭源 |
| GPT-2 | 通用 LM |
| Salesforce/codegen-350M-mono | 代码模型 |
| Salesforce/codegen2-7B | 代码模型 |
| bigcode/santacoder | 代码模型 |
| bigcode/starcoder | 代码模型 |
| microsoft/codebert-base | 代码理解模型 |
| Salesforce/codet5p-16b | 代码模型 |
| Salesforce/codet5-large | 代码模型 |
| replit/replit-code-v1_5-3b | 代码模型 |
| facebook/incoder-6B | 代码模型 |
| WizardLM/WizardCoder-Python-34B-V1.0 | 代码模型 |
| codellama/CodeLlama-7b-Python-hf | 代码模型 |
| deepseek-ai/deepseek-coder-6.7b-base | 代码模型 |

### 微调训练 & 评估模型

| 模型 | 参数规模 |
|:---|:---|
| Salesforce/codegen-350M-nl | 350M（主要实验模型） |
| EleutherAI/pythia-1b | 1B |
| TinyLlama/TinyLlama-1.1B | 1.1B |

### 评估 Benchmark

| Benchmark | 说明 |
|:---|:---|
| HumanEval（含自定义 `humaneval_spy` 变体） | 164 道编程题，pass@k |
| MBPP | 多选编程题 |

---

## TokenSugar

### 模式挖掘数据集（Pattern Mining）

| 数据集 | 来源 | 说明 |
|:---|:---|:---|
| LeetCode_Python_Solutions_v2 | `LimYeri/LeetCode_Python_Solutions_v2` (HuggingFace) | 用于挖掘高频 AST 子树模式 |

### 训练 / 推理数据集

| 数据集 | 说明 |
|:---|:---|
| StarCoderData (Python 子集) | 通过 `dataset_name` 参数指定，用于增量预训练 |

### Token 压缩率基准 Tokenizer

TokenSugar 的 miner 使用 **cl100k_base**（GPT-4 / ChatGPT tokenizer）作为基准 tokenizer 计算压缩收益。

### 推理评估模型

| 模型族 | Tokenizer 类型 | 代表模型 |
|:---|:---|:---|
| Yi-Coder | LlamaTokenizerFast | Yi 系列代码模型 |
| Pythia | GPTNeoXTokenizerFast | EleutherAI/pythia 系列 |
| Qwen | GPTNeoXTokenizerFast | Qwen 系列 |
| DeepSeek-Coder | LlamaTokenizerFast | deepseek-ai/deepseek-coder 系列 |
| Llama | PreTrainedTokenizerFast | Meta Llama 系列（含 CodeLlama） |

### 评估 Benchmark

| Benchmark | 说明 |
|:---|:---|
| HumanEval | 代码生成 |
| MBPP | 代码生成 |
| GSM8K (PAL) | 数学推理（用代码求解） |

---

## 核心对比

| 维度 | SimPy | TokenSugar |
|:---|:---|:---|
| 模式来源 | 预定义语法重写规则 | 从 LeetCode 挖掘高频 AST 子树 |
| 训练数据 | StarCoderData (100star Python) | StarCoderData (Python) |
| 压缩率测试模型 | 15 个主流 tokenizer | cl100k_base (GPT-4) 为基准 |
| 微调/训练模型 | CodeGen-350M, Pythia-1B, TinyLlama-1.1B | Yi-Coder, Pythia, Qwen, DeepSeek-Coder, Llama 系列 |
| 评估 Benchmark | HumanEval, MBPP | HumanEval, MBPP, GSM8K |
