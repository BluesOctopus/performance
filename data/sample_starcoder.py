import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except ImportError:
        return None

def extract_stars(ex):
    star_keys = ("max_stars_count", "max_stars_repo_stars", "stars", "repo_stars")
    for k in star_keys:
        if k in ex and ex[k] is not None:
            try:
                return int(ex[k])
            except:
                pass
    return 0

def main():
    parser = argparse.ArgumentParser(description="Sample code from StarCoderData for evaluation.")
    parser.add_argument("--lang", type=str, default="python", help="Language subset (default: python)")
    parser.add_argument("--min_stars", type=int, default=100, help="Minimum star count (default: 100)")
    parser.add_argument("--target_tokens", type=int, default=1_000_000, help="Target total tokens (default: 1,000,000)")
    parser.add_argument("--output_file", type=str, default="data/starcoder_sample.txt", help="Output text file")
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder", help="Tokenizer to use for counting")
    
    args = parser.parse_args()

    token = get_hf_token()
    
    # Load tokenizer
    print(f"Loading tokenizer {args.tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True, token=token)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to gpt2 tokenizer for estimation...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load dataset in streaming mode
    print(f"Loading starcoderdata (lang={args.lang}) in streaming mode...")
    try:
        ds = load_dataset("bigcode/starcoderdata", data_dir=args.lang, split="train", streaming=True, token=token)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have access to bigcode/starcoderdata and HF_TOKEN is set.")
        return

    total_tokens = 0
    sampled_code = []

    pbar = tqdm(total=args.target_tokens, desc="Sampling tokens")
    
    for entry in ds:
        # Check stars
        stars = extract_stars(entry)
        if stars <= args.min_stars:
            continue
            
        content = entry.get("content", "")
        if not content:
            continue
            
        # Count tokens
        # We use a simple estimation if tokenizer is slow, but for 1M tokens it should be fine
        tokens = tokenizer.encode(content)
        num_tokens = len(tokens)
        
        sampled_code.append(content)
        total_tokens += num_tokens
        pbar.update(num_tokens)
        
        if total_tokens >= args.target_tokens:
            break
            
    pbar.close()

    # Save to file
    output_path = os.path.join(os.getcwd(), args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving {len(sampled_code)} samples ({total_tokens} tokens) to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, code in enumerate(sampled_code):
            # Using a separator that is unlikely to be in the code
            f.write(f"<|sample_{i}|>\n")
            f.write(code)
            f.write("\n")

    print(f"Successfully sampled {total_tokens} tokens into {args.output_file}")

if __name__ == "__main__":
    main()

