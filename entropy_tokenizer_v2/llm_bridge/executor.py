import sys
import multiprocessing
import traceback
import json
from pathlib import Path
from tqdm import tqdm

def check_solution(task_id, generation, test, entry_point, results_dict):
    """在沙盒中执行代码并验证结果"""
    # 简单的后处理：提取代码块
    if "```python" in generation:
        code = generation.split("```python")[1].split("```")[0].strip()
    elif "```" in generation:
        # 找到第一个 ``` 之后的内容
        parts = generation.split("```")
        if len(parts) >= 3:
            code = parts[1].strip()
        else:
            code = generation.strip()
    else:
        code = generation.strip()

    # 构造完整的测试脚本
    # 注意：MBPP 的 test 已经是 assert 语句列表，不需要 check(entry_point)
    if entry_point:
        full_code = f"{code}\n\n{test}\n\ncheck({entry_point})"
    else:
        full_code = f"{code}\n\n{test}"
    
    # 使用 exec 执行代码，注意安全性
    try:
        # 限制全局变量
        exec_globals = {}
        # 执行
        exec(full_code, exec_globals)
        results_dict[task_id] = True
    except Exception:
        # print(f"Failed {task_id}: {traceback.format_exc()}")
        results_dict[task_id] = False

def calculate_pass_at_1(results_file):
    """计算 pass@1 指标"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    manager = multiprocessing.Manager()
    compressed_results = manager.dict()
    original_results = manager.dict()
    
    print(f"Starting execution for {len(data)} items...")
    
    # 分批处理，避免进程过多
    batch_size = 50
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        processes = []
        for item in batch:
            # 验证压缩版生成
            p1 = multiprocessing.Process(target=check_solution, args=(
                item['task_id'], item['compressed_generation'], item['test'], item.get('entry_point'), compressed_results
            ))
            # 验证原始版生成
            p2 = multiprocessing.Process(target=check_solution, args=(
                item['task_id'], item['original_generation'], item['test'], item.get('entry_point'), original_results
            ))
            processes.extend([p1, p2])
            p1.start()
            p2.start()

        for p in tqdm(processes, desc=f"Batch {i//batch_size + 1}", leave=False):
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    # 统计
    c_pass = sum(1 for v in compressed_results.values() if v)
    o_pass = sum(1 for v in original_results.values() if v)
    total = len(data)

    print("\n" + "="*40)
    print(f"  Benchmark Results: {Path(results_file).name} (n={total})")
    print("="*40)
    print(f"  Original (Baseline):   {o_pass}/{total} ({o_pass/total*100:.1f}%)" if total else "N/A")
    print(f"  Compressed (Proposed): {c_pass}/{total} ({c_pass/total*100:.1f}%)" if total else "N/A")
    print("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--tokenizer", type=str, default="deepseek")
    args = parser.parse_args()

    results_file = f"entropy_tokenizer_v2/llm_bridge/{args.benchmark}_test/{args.tokenizer}_eval_results.json"
    if Path(results_file).exists():
        calculate_pass_at_1(results_file)
    else:
        print(f"Results file not found: {results_file}")
