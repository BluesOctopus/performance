import sys
import os
import json
import re
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "eval"))

from repo_miner import _load_tokenizer, RepoConfig
from v2_eval import apply_v2_compression
from llm_bridge.prompt_builder import CompressionPromptBuilder

def load_humaneval_local():
    """
    尝试从本地或 HF 加载 HumanEval 数据集。
    为了演示，我们假设用户可以访问 openai_humaneval。
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        return ds
    except Exception as e:
        print(f"Error loading HumanEval: {e}")
        return []

def run_humaneval_compression_test(tokenizer_key="deepseek"):
    """
    对 HumanEval 题目进行压缩，并生成用于 LLM 评估的 Prompt。
    """
    # 1. 加载配置
    from config import EVAL_TOKENIZERS
    cfg = EVAL_TOKENIZERS.get(tokenizer_key)
    tokenizer, tok_type = _load_tokenizer(tokenizer_key, cfg)
    
    # 2. 加载 RepoConfig (使用 starcoder 训练出的通用规律)
    cache_dir = Path(__file__).parent.parent / "cache"
    # 修正缓存文件名匹配逻辑，确保能找到 deepseek 的缓存
    config_path = cache_dir / f"repo_config_{tokenizer_key}_eval_1000.json"
    
    if not config_path.exists():
        # 尝试另一种可能的命名格式
        config_path = cache_dir / f"repo_config_{tokenizer_key}_starcoderdata_1m_{tokenizer_key}.json"
    
    if not config_path.exists():
        print(f"RepoConfig not found in {cache_dir}. Please run v2_eval.py first.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        repo_config = RepoConfig.from_json(f.read())

    # 3. 加载 HumanEval
    import os
    from config import HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN
    dataset = load_humaneval_local()
    if not dataset:
        return

    output_dir = Path(__file__).parent / "humaneval_test"
    output_dir.mkdir(exist_ok=True)

    results = []
    
    print(f"Processing {len(dataset)} HumanEval problems...")
    for i, item in enumerate(tqdm(dataset)):
        task_id = item['task_id']
        prompt = item['prompt'] # 题目描述和函数头
        canonical_solution = item['canonical_solution']
        full_code = prompt + canonical_solution
        
        # 压缩完整代码（模拟模型需要理解的上下文）
        compressed_text, stats = apply_v2_compression(full_code, repo_config, tokenizer, tok_type)
        
        # 生成 Prompt
        builder = CompressionPromptBuilder(
            selected_skeletons=repo_config.selected_skeletons,
            replacement_map=repo_config.replacement_map
        )
        
        # 构造评估用的 Prompt：
        # 我们给模型看压缩后的 prompt 部分，要求它补全剩下的部分
        compressed_prompt, _ = apply_v2_compression(prompt, repo_config, tokenizer, tok_type)
        
        llm_input = builder.generate_full_prompt(compressed_prompt)
        llm_input += "\n\n### Task:\nComplete the Python function above. Return ONLY the decompressed Python code."

        # 保存结果
        problem_data = {
            "task_id": task_id,
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "llm_input_prompt": llm_input,
            "canonical_solution": canonical_solution,
            "test": item['test'],
            "entry_point": item['entry_point']
        }
        results.append(problem_data)

    # 保存所有题目到 JSON，方便后续调用 LLM
    output_file = output_dir / f"humaneval_compressed_{tokenizer_key}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccess! Generated {len(results)} compressed HumanEval problems.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    run_humaneval_compression_test()
