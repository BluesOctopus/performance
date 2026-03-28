import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
import math
from tqdm import tqdm
import json
from pathlib import Path

def compute_perplexity(model, tokenizer, text, device):
    """Compute perplexity of a text under a given model and tokenizer."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # HF models usually have a max length
    if input_ids.size(1) > 1024:
        input_ids = input_ids[:, :1024]
        
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    return loss.item(), input_ids.size(1)

def run_loss_probing(model_id="gpt2", num_samples=50):
    print(f"--- Starting Loss Probing (Model: {model_id}, Samples: {num_samples}) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Base Model and Tokenizer
    from config import HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN
    print(f"Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN, low_cpu_mem_usage=True).to(device)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    print("Model loaded successfully.")
    model.eval()
    
    # 2. Load NS Tokenizer (Stage 1 is minimal, Stage 2/3 are active)
    # We use the same samples to mine skeletons for Stage 1 to be fair
    samples = load_eval_samples(num_samples + 50) # Extra for mining
    mining_samples = samples[:50]
    eval_samples = samples[50:50+num_samples]
    
    from repo_miner import mine_from_sources
    from config import EVAL_TOKENIZERS
    tok_key = "gpt2" if "gpt2" in model_id else "gpt4"
    tok_cfg = EVAL_TOKENIZERS.get(tok_key, EVAL_TOKENIZERS["gpt2"])
    
    repo_config = mine_from_sources(mining_samples, tok_key, tok_cfg, cache=False, verbose=False)
    ns_tokenizer = NeuralSymbolicTokenizer(tokenizer_key=tok_key, repo_config=repo_config)
    
    results = []
    total_loss_orig = 0
    total_tokens_orig = 0
    total_loss_comp = 0
    total_tokens_comp = 0
    
    for i, code in enumerate(tqdm(eval_samples, desc="Probing")):
        # Original Loss
        loss_orig, n_orig = compute_perplexity(model, base_tokenizer, code, device)
        
        # Compressed Loss (NS Tokenizer)
        output = ns_tokenizer.encode(code)
        compressed_text = output.text
        loss_comp, n_comp = compute_perplexity(model, base_tokenizer, compressed_text, device)
        
        results.append({
            "id": i,
            "orig_loss": loss_orig,
            "orig_tokens": n_orig,
            "comp_loss": loss_comp,
            "comp_tokens": n_comp,
            "compression_ratio": n_comp / n_orig if n_orig > 0 else 1.0
        })
        
        total_loss_orig += loss_orig * n_orig
        total_tokens_orig += n_orig
        total_loss_comp += loss_comp * n_comp
        total_tokens_comp += n_comp
        
    avg_loss_orig = total_loss_orig / total_tokens_orig if total_tokens_orig > 0 else 0
    avg_loss_comp = total_loss_comp / total_tokens_comp if total_tokens_comp > 0 else 0
    
    ppl_orig = math.exp(avg_loss_orig)
    ppl_comp = math.exp(avg_loss_comp)
    
    print("\n" + "="*50)
    print(f"LOSS PROBING RESULTS")
    print("="*50)
    print(f"Original Avg Loss: {avg_loss_orig:.4f} (PPL: {ppl_orig:.2f})")
    print(f"Compressed Avg Loss: {avg_loss_comp:.4f} (PPL: {ppl_comp:.2f})")
    print(f"Token Reduction: {(1 - total_tokens_comp/total_tokens_orig)*100:.1f}%")
    print("="*50)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "loss_probing_results.json", "w") as f:
        json.dump({
            "summary": {
                "avg_loss_orig": avg_loss_orig,
                "ppl_orig": ppl_orig,
                "avg_loss_comp": avg_loss_comp,
                "ppl_comp": ppl_comp,
                "token_reduction": 1 - total_tokens_comp/total_tokens_orig
            },
            "details": results
        }, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--num_samples", type=int, default=30)
    args = parser.parse_args()
    
    run_loss_probing(model_id=args.model, num_samples=args.num_samples)
