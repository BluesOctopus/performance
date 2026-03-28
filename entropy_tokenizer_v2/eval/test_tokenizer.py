"""Test script to verify 100% reversibility of NeuralSymbolicTokenizer 
on a variety of code samples.
"""

import sys
import os
import ast

# Add project root to path
current_dir = os.path.abspath(os.path.dirname(__file__)) # eval dir
parent_dir = os.path.dirname(current_dir) # project root
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tokenizer import NeuralSymbolicTokenizer
from eval.v2_eval import load_eval_samples
from repo_miner import mine_from_sources

def are_semantically_equivalent(code1, code2):
    """Check if two code snippets are semantically equivalent."""
    # 1. Try AST comparison (best)
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        return ast.dump(tree1) == ast.dump(tree2)
    except Exception:
        pass
        
    # 2. Fallback: Compare non-blank lines (ignoring indentation changes if not parsable)
    # Note: This is a weak check, but better than nothing for snippets.
    lines1 = [line.strip() for line in code1.splitlines() if line.strip()]
    lines2 = [line.strip() for line in code2.splitlines() if line.strip()]
    return lines1 == lines2

def test_reversibility(tokenizer_key="gpt4", num_samples=10):
    print(f"Testing reversibility (Semantic) for {tokenizer_key} on {num_samples} samples...")
    
    # 1. Load samples
    samples = load_eval_samples(num_samples)
    
    # 2. Mine Stage 1 config
    from config import EVAL_TOKENIZERS
    cfg = mine_from_sources(samples, tokenizer_key, EVAL_TOKENIZERS[tokenizer_key], cache=False, verbose=False)
    
    # 3. Init Tokenizer
    ns_tokenizer = NeuralSymbolicTokenizer(tokenizer_key=tokenizer_key, repo_config=cfg)
    
    # 4. Test each sample
    success_count = 0
    for i, code in enumerate(samples):
        try:
            output = ns_tokenizer.encode(code)
            restored = ns_tokenizer.decode(output.text)
            
            if are_semantically_equivalent(code, restored):
                success_count += 1
            else:
                print(f"Sample {i} FAILED Semantic Equivalence!")
                import difflib
                try:
                    diff = difflib.unified_diff(
                        code.strip().splitlines(),
                        restored.strip().splitlines(),
                        fromfile='original',
                        tofile='restored'
                    )
                    # Use utf-8 for printing if possible
                    diff_text = '\n'.join(diff)
                    print(diff_text.encode('ascii', 'replace').decode('ascii'))
                except Exception as e:
                    print(f"  Could not print diff: {e}")
                # break # Show first failure
        except Exception as e:
            print(f"Sample {i} CRASHED: {e}")
            
    print(f"Semantic Reversibility: {success_count}/{len(samples)} ({(success_count/len(samples))*100:.1f}%)")
    return success_count == len(samples)

if __name__ == "__main__":
    t_key = "gpt4"
    if len(sys.argv) > 1:
        t_key = sys.argv[1]
        
    ok = test_reversibility(tokenizer_key=t_key, num_samples=20)
    sys.exit(0 if ok else 1)
