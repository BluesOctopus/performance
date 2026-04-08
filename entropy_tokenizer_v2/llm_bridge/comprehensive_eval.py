import sys
import os
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "eval"))

from repo_miner import _load_tokenizer, RepoConfig
from v2_eval import apply_v2_compression
from llm_bridge.prompt_builder import CompressionPromptBuilder
from marker_count import count_augmented, RE_ALL_MARKERS
from config import EVAL_TOKENIZERS

def get_token_count_native(text, tokenizer, tok_type):
    """原生 Tokenizer 统计 (会切分自定义标记)"""
    if tok_type == "tiktoken":
        return len(tokenizer.encode(text))
    else:
        return len(tokenizer.encode(text, add_special_tokens=False))

def get_token_count_augmented(text, tokenizer, tok_type):
    """理想 Tokenizer 统计 (自定义标记计为 1)"""
    return count_augmented(text, tokenizer, tok_type, pattern=RE_ALL_MARKERS)

def run_comprehensive_length_test(tokenizer_key="deepseek"):
    """
    全维度长度梯度测试：分析不同文件长度下的压缩收益曲线。
    """
    # 1. 加载配置
    cfg = EVAL_TOKENIZERS.get(tokenizer_key)
    tokenizer, tok_type = _load_tokenizer(tokenizer_key, cfg)
    
    # 2. 加载 RepoConfig
    cache_dir = Path(__file__).parent.parent / "cache"
    config_path = cache_dir / f"repo_config_{tokenizer_key}_eval_1000.json"
    if not config_path.exists():
        config_path = cache_dir / f"repo_config_{tokenizer_key}_starcoderdata_1m_{tokenizer_key}.json"
    
    if not config_path.exists():
        print(f"RepoConfig not found for {tokenizer_key}. Please run v2_eval.py first.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        repo_config = RepoConfig.from_json(f.read())

    # 3. 加载数据并按长度分组
    data_file = Path("C:/Users/25818/Desktop/SITP/paper/performance/code/data/starcoder_1m_tokens.txt")
    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    samples = [p.strip() for p in re.split(r'<\|sample_\d+\|>', content) if p.strip()]
    
    # 定义长度梯度 (字符数)
    bins = [
        (0, 500, "Very Short"),
        (500, 2000, "Short"),
        (2000, 5000, "Medium"),
        (5000, 15000, "Long"),
        (15000, 50000, "Very Long")
    ]
    
    results_by_bin = {name: [] for _, _, name in bins}
    
    print(f"Analyzing {len(samples)} samples across length gradients...")
    
    for src in tqdm(samples):
        char_len = len(src)
        bin_name = None
        for low, high, name in bins:
            if low <= char_len < high:
                bin_name = name
                break
        if not bin_name: continue
        if len(results_by_bin[bin_name]) >= 20: continue # 每组取20个样本平衡数据

        # 压缩
        compressed_text, _ = apply_v2_compression(src, repo_config, tokenizer, tok_type)
        
        # 生成动态说明书 (Dynamic Dictionary)
        builder = CompressionPromptBuilder(
            selected_skeletons=repo_config.selected_skeletons,
            replacement_map=repo_config.replacement_map
        )
        full_prompt = builder.generate_full_prompt(compressed_text)
        
        # 统计数据
        # 1. 原始 Token (Native)
        orig_tokens = get_token_count_native(src, tokenizer, tok_type)
        
        # 2. 压缩后代码 Token (Native - 现有的 API 模式)
        code_part = full_prompt.split("--- COMPRESSED CODE ---")[-1]
        comp_code_tokens_native = get_token_count_native(code_part, tokenizer, tok_type)
        
        # 3. 压缩后代码 Token (Augmented - 理想词表注入模式)
        comp_code_tokens_aug = get_token_count_augmented(compressed_text, tokenizer, tok_type)
        
        # 4. 说明书开销 (Native)
        overhead_tokens = get_token_count_native(full_prompt, tokenizer, tok_type) - comp_code_tokens_native
        
        # 计算节省率
        # API 模式节省率 (含说明书)
        api_saving = (orig_tokens - (comp_code_tokens_native + overhead_tokens)) / orig_tokens * 100
        # 理想模式节省率 (含说明书，假设说明书也注入词表或均摊)
        ideal_saving = (orig_tokens - (comp_code_tokens_aug + overhead_tokens)) / orig_tokens * 100
        # 纯代码理想压缩率 (不含说明书)
        pure_code_saving = (orig_tokens - comp_code_tokens_aug) / orig_tokens * 100

        results_by_bin[bin_name].append({
            "char_len": char_len,
            "api_saving": api_saving,
            "ideal_saving": ideal_saving,
            "pure_code_saving": pure_code_saving,
            "overhead": overhead_tokens
        })

    # 输出报表
    print("\n" + "="*85)
    print(f"{'Length Category':<15} | {'Avg Chars':>10} | {'Overhead':>8} | {'API%':>8} | {'Ideal%':>8} | {'Pure%':>8}")
    print("-" * 85)
    
    summary_data = []
    for _, _, name in bins:
        data = results_by_bin[name]
        if not data: continue
        avg_chars = sum(d['char_len'] for d in data) / len(data)
        avg_overhead = sum(d['overhead'] for d in data) / len(data)
        avg_api = sum(d['api_saving'] for d in data) / len(data)
        avg_ideal = sum(d['ideal_saving'] for d in data) / len(data)
        avg_pure = sum(d['pure_code_saving'] for d in data) / len(data)
        
        print(f"{name:<15} | {avg_chars:>10.0f} | {avg_overhead:>8.0f} | {avg_api:>7.1f}% | {avg_ideal:>7.1f}% | {avg_pure:>7.1f}%")
        summary_data.append((avg_chars, avg_api, avg_ideal, avg_pure))

    print("="*85)
    print("API%: Current API (Native Tokenizer + Overhead)")
    print("Ideal%: Augmented Tokenizer + Overhead")
    print("Pure%: Augmented Tokenizer (No Overhead, e.g. Repository-level)")

if __name__ == "__main__":
    run_comprehensive_length_test()
