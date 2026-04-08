import asyncio
import json
import time
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# 配置
API_KEY = "sk-d82ec82b33534a039c43b25e7516d573"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat" # 或者 deepseek-coder
CONCURRENT_REQUESTS = 10 # DeepSeek 接口通常支持更高的并发

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

async def call_llm_async(prompt, task_id, mode):
    """异步调用 LLM"""
    for attempt in range(3):
        try:
            # 修正：直接在 client 上调用 with_options，或者直接使用 timeout 参数
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                timeout=60.0 # 增加超时时间
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                print(f"\n[Error] {task_id} ({mode}) failed after 3 attempts: {e}")
            await asyncio.sleep(2 * (attempt + 1))
    return None

async def process_task(prob, semaphore, results, completed_ids):
    """处理单个 HumanEval 任务"""
    if prob['task_id'] in completed_ids:
        return

    async with semaphore:
        # 1. 测试压缩版
        compressed_task = call_llm_async(prob['llm_input_prompt'], prob['task_id'], "Compressed")
        
        # 2. 测试原始版
        original_prompt = f"Complete the following Python function:\n\n```python\n{prob['original_prompt']}\n```\nReturn ONLY the completed code."
        original_task = call_llm_async(original_prompt, prob['task_id'], "Original")
        
        # 并行执行这两个请求
        compressed_res, original_res = await asyncio.gather(compressed_task, original_task)
        
        if compressed_res and original_res:
            results.append({
                "task_id": prob['task_id'],
                "compressed_generation": compressed_res,
                "original_generation": original_res,
                "entry_point": prob.get('entry_point'), # 使用 get 避免 KeyError
                "test": prob['test']
            })

async def run_eval_async(input_json_path, output_json_path):
    if not Path(input_json_path).exists():
        print(f"Input file not found: {input_json_path}")
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    results = []
    if Path(output_json_path).exists():
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            pass
    
    completed_ids = {r['task_id'] for r in results}
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    tasks = []
    for prob in problems:
        tasks.append(process_task(prob, semaphore, results, completed_ids))
    
    print(f"Starting Async Evaluation for {Path(input_json_path).name} (Concurrency={CONCURRENT_REQUESTS})...")
    # 使用 tqdm 显示进度
    await tqdm.gather(*tasks)

    # 最终保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation complete. Results saved to {output_json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--tokenizer", type=str, default="deepseek")
    args = parser.parse_args()

    input_path = f"entropy_tokenizer_v2/llm_bridge/{args.benchmark}_test/{args.benchmark}_compressed_{args.tokenizer}.json"
    output_path = f"entropy_tokenizer_v2/llm_bridge/{args.benchmark}_test/{args.tokenizer}_eval_results.json"
    
    asyncio.run(run_eval_async(input_path, output_path))
