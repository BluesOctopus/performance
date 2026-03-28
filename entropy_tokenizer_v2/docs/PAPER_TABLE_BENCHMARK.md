# 论文表 tokenizer × Starcoder 约 1M 语料（v2 全链路）

- 样本：`performance/code/data/starcoder_1m_tokens.txt`（301 段）。
- 挖掘前 **保留** `#` 注释与 docstring（`lossless_clean`）。
- 表为 **v2 三阶段** 相对 baseline token 的降幅；与论文中 SimPy 列含义不同。

| Tokenizer | Vocab 来源 | V₀ | Baseline tokens | Final tokens | 降幅 |
|-----------|------------|----|-----------------|--------------|------|
| CodeBERT | Code | 50,265 | 1,002,719 | 515,466 | 48.59% |
| GPT-2 | Web | 50,257 | 1,002,719 | 515,466 | 48.59% |
| CodeLlama | Web | 32,016 | 704,587 | 510,281 | 27.58% |
| DeepSeek-Coder | Web | 32,000 | 699,738 | 507,059 | 27.54% |
| CodeGen | Web | 50,257 | 689,052 | 510,358 | 25.93% |
| Codex | Web | 50,257 | 689,052 | 510,358 | 25.93% |
| SantaCoder | Code | 49,152 | 593,452 | 478,856 | 19.31% |
| GPT-3.5 | Web | 100,277 | 521,319 | 405,062 | 22.30% |
| GPT-4 | Web | 100,277 | 521,319 | 405,062 | 22.30% |

## 跳过或失败

- `wizardcoder`: OSError: WizardLM/WizardCoder-Python-7B-V1.0 is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
- `codet5p`: TypeError: Input must be a List[Union[str, AddedToken]]
- `codet5`: TypeError: Input must be a List[Union[str, AddedToken]]
- `starcoder`: OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/bigcode/starcoder.
401 Client Error. (Request ID: Root=1-69bfb917-2ec38c9b154411da22b725da;4d3dab9d-cd2d-49db-bfa8-1cef78f3f6f0)

Cannot access gated repo for url https://huggingface.co/bigcode/starcoder/resolve/main/config.json.
Access to model bigcode/starcoder is restricted. You must have access to it and be authenticated to access it. Please log in.
- `replit-code`: ImportError: This modeling file requires the following packages that were not found in your environment: sentencepiece. Run `pip install sentencepiece`
