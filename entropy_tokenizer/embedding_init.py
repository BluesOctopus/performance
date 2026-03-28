import torch
import json
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer

from config import CACHE_DIR

def load_dynamic_operators():
    """
    Load dynamically mined operators from the cache.
    Returns a list of dicts with operator metadata.
    """
    cache_path = CACHE_DIR / "mining_results.json"
    if not cache_path.exists():
        print(f"[Embedding Init] Warning: No dynamic mining results found at {cache_path}")
        return []
        
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        ops = []
        
        # Format AST skeletons into Operator-like structs
        ast_patterns = data.get("ast_patterns", {})
        for i, skeleton in enumerate(ast_patterns.keys()):
            ops.append({
                "type": "syntax",
                "token_str": f"<OP_AST_{i}>",
                "skeleton": skeleton,
                "text_repr": skeleton.replace("{0}", "").replace("{1}", "").replace("{2}", "").replace("{3}", "").strip()
            })
            
        # Format Lexical prefixes
        lexical_prefixes = data.get("lexical_prefixes", {})
        for i, prefix in enumerate(lexical_prefixes.keys()):
            ops.append({
                "type": "lexical",
                "token_str": f"<OP_PRE_{i}>",
                "text_repr": prefix
            })
            
        return ops
    except Exception as e:
        print(f"[Embedding Init] Error loading dynamic operators: {e}")
        return []

def smart_initialize_embeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dynamic_ops=None):
    """
    Initializes embeddings for newly added operator tokens using Mean Pooling of their constituent tokens.
    
    1. model.resize_token_embeddings(len(tokenizer)) must have been called already,
       or this function can handle it. We assume tokenizer already has the new tokens.
    2. We fetch the embedding matrix.
    3. For each operator token, we find its representative string, encode it with the base tokenizer,
       and take the mean of the resulting token embeddings.
    """
    # Resize embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    embeddings = model.get_input_embeddings()
    weight = embeddings.weight.data
    
    if dynamic_ops is None:
        dynamic_ops = load_dynamic_operators()
        
    if not dynamic_ops:
        print("[Embedding Init] No operators provided for initialization.")
        return model
    
    with torch.no_grad():
        for op in dynamic_ops:
            token_str = op["token_str"]
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            
            # If the token was not added for some reason, skip
            if token_id == tokenizer.unk_token_id and token_str != tokenizer.unk_token:
                continue
                
            text_repr = op.get("text_repr", "")
            if not text_repr:
                continue
                
            # Encode the representative text using the tokenizer
            # We don't want to trigger the operator encoding, so we use raw text
            constituent_ids = tokenizer.encode(text_repr, add_special_tokens=False)
            
            # If it resolves to some tokens, take the mean
            if constituent_ids:
                # Filter out UNK tokens from constituents if possible
                valid_ids = [idx for idx in constituent_ids if idx != tokenizer.unk_token_id]
                if not valid_ids:
                    valid_ids = constituent_ids
                    
                vectors = weight[valid_ids]
                mean_vec = vectors.mean(dim=0)
                
                # Assign the mean vector to the new operator token
                weight[token_id] = mean_vec
                print(f"[Embedding Init] Initialized {token_str} using mean of: {tokenizer.convert_ids_to_tokens(valid_ids)}")
            else:
                print(f"[Embedding Init] Warning: No constituent tokens found for {token_str} ('{text_repr}')")

    return model
