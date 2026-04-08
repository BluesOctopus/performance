import json
import tiktoken
from pathlib import Path

def analyze_costs(input_json_path, tokenizer_name="gpt-4"):
    """
    分析 HumanEval 实验中每一道题的 Token 消耗。
    """
    # 使用 tiktoken 估算 GPT-4/DeepSeek 的 token 数 (两者分词器长度接近)
    enc = tiktoken.encoding_for_model("gpt-4")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    total_original = 0
    total_compressed_code = 0
    total_overhead = 0
    total_net_input = 0
    
    print(f"{'Task ID':<20} | {'Original':>8} | {'CompCode':>8} | {'Overhead':>8} | {'Net Saving':>10}")
    print("-" * 65)
    
    for prob in problems:
        # 1. 原始 Prompt Token
        original_prompt = f"Complete the following Python function:\n\n```python\n{prob['original_prompt']}\n```\nReturn ONLY the completed code."
        n_original = len(enc.encode(original_prompt))
        
        # 2. 压缩版总 Input Token
        n_full_input = len(enc.encode(prob['llm_input_prompt']))
        
        # 3. 提取压缩版中的代码部分 (估算)
        # 找到 "--- COMPRESSED CODE ---" 之后的部分
        code_part = prob['llm_input_prompt'].split("--- COMPRESSED CODE ---")[-1]
        n_comp_code = len(enc.encode(code_part))
        
        # 4. 计算说明书开销 (Overhead)
        n_overhead = n_full_input - n_comp_code
        
        # 5. 净节省 (针对单次请求)
        net_saving = n_original - n_full_input
        
        total_original += n_original
        total_compressed_code += n_comp_code
        total_overhead += n_overhead
        total_net_input += n_full_input
        
        # 打印前 10 个作为示例
        if problems.index(prob) < 10:
            print(f"{prob['task_id']:<20} | {n_original:>8} | {n_comp_code:>8} | {n_overhead:>8} | {net_saving:>10}")

    print("-" * 65)
    print(f"{'TOTAL (164 tasks)':<20} | {total_original:>8,} | {total_compressed_code:>8,} | {total_overhead:>8,} | {total_original - total_net_input:>10,}")
    
    print("\n结论分析:")
    code_reduction = (1 - total_compressed_code / total_original) * 100
    print(f"1. 纯代码压缩率 (Pure Code Reduction): {code_reduction:.1f}%")
    print(f"2. 平均每题说明书开销 (Avg Overhead per task): {total_overhead/len(problems):.1f} tokens")
    
    if total_original > total_net_input:
        print(f"3. 最终结果: 节省了 {total_original - total_net_input} tokens (正收益!)")
    else:
        print(f"3. 最终结果: 额外花费了 {total_net_input - total_original} tokens (负收益)")
        print("   原因: 对于 HumanEval 这种短代码题目，说明书的固定开销超过了代码节省量。")
        print("   论文建议: 强调在『长文本/多文件/Repository 级别』任务中，说明书只发一次，收益将变为正。")

if __name__ == "__main__":
    # 使用生成好的压缩题目文件进行分析
    input_path = "entropy_tokenizer_v2/llm_bridge/humaneval_test/humaneval_compressed_deepseek.json"
    analyze_costs(input_path)
