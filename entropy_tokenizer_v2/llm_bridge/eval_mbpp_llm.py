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

def load_mbpp_local():
    """
    加载 MBPP 数据集。
    """
    try:
        from datasets import load_dataset
        # 使用 'mbpp' 数据集
        ds = load_dataset("mbpp", "full", split="test")
        return ds
    except Exception as e:
        print(f"Error loading MBPP: {e}")
        return []

def run_mbpp_compression_test(tokenizer_key="deepseek"):
    """
    对 MBPP 题目进行压缩，并生成用于 LLM 评估的 Prompt。
    """
    # 1. 加载配置
    from config import EVAL_TOKENIZERS, HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN
    cfg = EVAL_TOKENIZERS.get(tokenizer_key)
    tokenizer, tok_type = _load_tokenizer(tokenizer_key, cfg)
    
    # 2. 加载 RepoConfig
    cache_dir = Path(__file__).parent.parent / "cache"
    config_path = cache_dir / f"repo_config_{tokenizer_key}_eval_1000.json"
    
    if not config_path.exists():
        config_path = cache_dir / f"repo_config_{tokenizer_key}_starcoderdata_1m_{tokenizer_key}.json"
    
    if not config_path.exists():
        print(f"RepoConfig not found. Please run v2_eval.py first.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        repo_config = RepoConfig.from_json(f.read())

    # 3. 加载 MBPP
    dataset = load_mbpp_local()
    if not dataset:
        return

    output_dir = Path(__file__).parent / "mbpp_test"
    output_dir.mkdir(exist_ok=True)

    results = []
    
    print(f"Processing {len(dataset)} MBPP problems...")
    for i, item in enumerate(tqdm(dataset)):
        task_id = item['task_id']
        # MBPP 的 prompt 是 'text' (自然语言描述)
        # 还有 'code' (参考代码) 和 'test_list' (单元测试)
        text_desc = item['text']
        test_code = "\n".join(item['test_list'])
        
        # 构造一个类似 HumanEval 的 Prompt：描述 + 函数签名
        # MBPP 原始数据里没有直接的函数签名，通常需要从参考代码里提取
        # 这里我们直接用描述作为 Prompt
        prompt = f'"""\n{text_desc}\n"""\n'
        
        # 压缩 Prompt
        compressed_prompt, _ = apply_v2_compression(prompt, repo_config, tokenizer, tok_type)
        
        # 生成 Prompt
        builder = CompressionPromptBuilder(
            selected_skeletons=repo_config.selected_skeletons,
            replacement_map=repo_config.replacement_map
        )
        
        llm_input = builder.generate_full_prompt(compressed_prompt)
        llm_input += "\n\n### Task:\nWrite the Python function described above. Return ONLY the decompressed Python code."

        # 保存结果
        problem_data = {
            "task_id": task_id,
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "llm_input_prompt": llm_input,
            "test": test_code,
            "canonical_solution": item['code']
        }
        results.append(problem_data)

    # 保存
    output_file = output_dir / f"mbpp_compressed_{tokenizer_key}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccess! Generated {len(results)} compressed MBPP problems.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    run_mbpp_compression_test()
