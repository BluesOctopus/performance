import ast
import json
import random
from tqdm import tqdm
from pathlib import Path
import sys
import os
import re

# 确保能导入项目模块
current_dir = os.path.abspath(os.path.dirname(__file__)) # eval dir
parent_dir = os.path.dirname(current_dir) # project root
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tokenizer import NeuralSymbolicTokenizer
from eval.v2_eval import load_eval_samples
from repo_miner import mine_from_sources

def clean_sample(code):
    """Strip <filename> headers and other non-Python markers."""
    code = re.sub(r'^<filename>.*?\n', '', code)
    return code.strip()

def compare_asts(code1, code2):
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        return ast.dump(tree1) == ast.dump(tree2)
    except Exception as e:
        return f"AST Parse Error: {e}"

def run_stress_test(num_samples=1000, tokenizer_key="gpt4"):
    print(f"--- Starting AST Stress Test ({num_samples} samples, {tokenizer_key}) ---")
    
    # 1. 加载并清洗样本
    raw_samples = load_eval_samples(num_samples)
    samples = []
    for s in raw_samples:
        cleaned = clean_sample(s)
        try:
            ast.parse(cleaned)
            samples.append(cleaned)
        except:
            continue
            
    print(f"Loaded {len(samples)} valid Python samples (filtered from {len(raw_samples)}).")
    if not samples:
        return

    # 2. 挖掘 Stage 1 骨架
    print("Mining skeletons for Stage 1...")
    from config import EVAL_TOKENIZERS
    tok_cfg = EVAL_TOKENIZERS[tokenizer_key]
    repo_config = mine_from_sources(samples[:100], tokenizer_key, tok_cfg, cache=False, verbose=False)
    
    # 3. 初始化 Tokenizer
    ns_tokenizer = NeuralSymbolicTokenizer(tokenizer_key=tokenizer_key, repo_config=repo_config)
    
    success_count = 0
    fail_count = 0
    parse_errors = 0
    semantic_mismatches = 0
    
    results = []

    for i, original_code in enumerate(tqdm(samples, desc="Testing")):
        try:
            # Encode
            output = ns_tokenizer.encode(original_code)
            compressed = output.text
            
            # Decode
            restored = ns_tokenizer.decode(compressed)
            
            # Compare
            comparison = compare_asts(original_code, restored)
            
            if comparison is True:
                success_count += 1
            else:
                fail_count += 1
                if "AST Parse Error" in str(comparison):
                    parse_errors += 1
                else:
                    semantic_mismatches += 1
                
                # 记录失败案例
                results.append({
                    "id": i,
                    "error": str(comparison),
                    "original": original_code,
                    "compressed": compressed,
                    "restored": restored
                })
                
        except Exception as e:
            fail_count += 1
            results.append({
                "id": i,
                "error": f"Runtime Exception: {e}",
                "original": original_code
            })

    # 4. 汇报结果
    print("\n" + "="*50)
    print(f"STRESS TEST RESULTS")
    print("="*50)
    print(f"Total Valid Samples: {len(samples)}")
    print(f"Success:            {success_count} ({(success_count/len(samples))*100:.1f}%)")
    print(f"Failures:           {fail_count}")
    print(f"  - Parse Errors:   {parse_errors}")
    print(f"  - Semantic Mismatch: {semantic_mismatches}")
    print("="*50)

    if results:
        results_dir = Path(parent_dir) / "results"
        results_dir.mkdir(exist_ok=True)
        error_log = results_dir / "stress_test_errors.json"
        with open(error_log, "w", encoding="utf-8") as f:
            json.dump(results[:10], f, indent=2, ensure_ascii=False)
        print(f"Error details (first 10) saved to: {error_log}")

if __name__ == "__main__":
    run_stress_test(num_samples=301)
