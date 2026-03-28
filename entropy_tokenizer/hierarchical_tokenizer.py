import re
from typing import List, Union
from transformers import PreTrainedTokenizer

class HierarchicalTokenizer:
    """
    A wrapper around a HuggingFace BPE tokenizer that implements Operator-Based Hierarchical Tokenization.
    
    Encoding:
      1. Applies Regex/AST-based substitutions to map code patterns to operator tokens.
      2. Passes the modified text to the base BPE tokenizer.
      
    Decoding:
      1. Decodes BPE tokens to text.
      2. Reconstructs the original code by applying the operator templates.
    """
    
    def __init__(self, base_tokenizer: PreTrainedTokenizer, dynamic_ops: list = None):
        self.base_tokenizer = base_tokenizer
        self.dynamic_ops = dynamic_ops or []
        
        # Add all operator tokens and special markers to the base tokenizer
        all_ops = [op["token_str"] for op in self.dynamic_ops]
        
        # Special markers to delineate arguments, preventing BPE from mixing them up
        self.arg_sep = "\ue000"
        self.arg_end = "\ue001"
        self.base_tokenizer.add_tokens(all_ops + [self.arg_sep, self.arg_end])
        
        # Build regex patterns for encoding and decoding
        self._build_regex_patterns()

    def _build_regex_patterns(self):
        self.encode_patterns = []
        self.decode_patterns = []
        
        for op in self.dynamic_ops:
            token_str = op["token_str"]
            
            if op["type"] == "syntax":
                skeleton = op["skeleton"]
                
                # Build encode regex: escape the skeleton, but replace {0}, {1} etc. with capture groups
                # Example skeleton: for {0} in range({1}):
                # Escaped: for\ \{0\}\ in\ range\(\{1\}\)\:
                enc_pattern = re.escape(skeleton)
                enc_pattern = enc_pattern.replace(r'\{0\}', r'(.*?)')
                enc_pattern = enc_pattern.replace(r'\{1\}', r'(.*?)')
                enc_pattern = enc_pattern.replace(r'\{2\}', r'(.*?)')
                enc_pattern = enc_pattern.replace(r'\{3\}', r'(.*?)')
                
                # Make whitespace flexible
                enc_pattern = re.sub(r'\\\s+', r'\\s+', enc_pattern)
                
                # Determine how many arguments we have
                num_args = skeleton.count("{")
                
                if num_args == 0:
                    enc_replacement = token_str
                    dec_pattern = re.escape(token_str)
                    dec_replacement = skeleton
                else:
                    # Build replacement string: <OP> \1 \ue000 \2 \ue000 \3 \ue001
                    repl_parts = [token_str]
                    for i in range(1, num_args + 1):
                        repl_parts.append(f'\\{i}')
                        if i < num_args:
                            repl_parts.append(self.arg_sep)
                    repl_parts.append(self.arg_end)
                    enc_replacement = ' '.join(repl_parts)
                    
                    # Build decode regex
                    dec_parts = [re.escape(token_str)]
                    for i in range(num_args):
                        dec_parts.append(r'\s*(.*?)\s*')
                        if i < num_args - 1:
                            dec_parts.append(re.escape(self.arg_sep))
                    dec_parts.append(re.escape(self.arg_end))
                    dec_pattern = ''.join(dec_parts)
                    
                    # Build decode replacement
                    dec_replacement = skeleton
                    for i in range(num_args):
                        dec_replacement = dec_replacement.replace(f"{{{i}}}", f"\\{i+1}")
                
                self.encode_patterns.append((re.compile(enc_pattern), enc_replacement))
                self.decode_patterns.append((re.compile(dec_pattern), dec_replacement))

            elif op["type"] == "lexical":
                prefix = op["text_repr"]
                
                # Encode: get_apple -> <OP> apple
                enc_pattern = rf'\b{re.escape(prefix)}([a-zA-Z0-9_]+)\b'
                enc_replacement = f'{token_str} \\1'
                self.encode_patterns.append((re.compile(enc_pattern), enc_replacement))
                
                # Decode: <OP> apple -> get_apple
                dec_pattern = rf'{re.escape(token_str)}\s*([a-zA-Z0-9_]+)'
                dec_replacement = rf'{prefix}\1'
                self.decode_patterns.append((re.compile(dec_pattern), dec_replacement))

    def pre_tokenize(self, text: str) -> str:
        """Apply structural and lexical transformations to the text."""
        encoded_text = text
        for pattern, replacement in self.encode_patterns:
            encoded_text = pattern.sub(replacement, encoded_text)
        return encoded_text

    def post_tokenize(self, text: str) -> str:
        """Reconstruct the original text from the decoded intermediate string."""
        decoded_text = text
        for pattern, replacement in self.decode_patterns:
            decoded_text = pattern.sub(replacement, decoded_text)
            
        # Clean up any leftover special markers just in case
        decoded_text = decoded_text.replace(self.arg_sep, "").replace(self.arg_end, "")
        return decoded_text

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode code text to a list of token IDs."""
        transformed = self.pre_tokenize(text)
        return self.base_tokenizer.encode(transformed, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode a list of token IDs back to code."""
        intermediate_text = self.base_tokenizer.decode(token_ids, **kwargs)
        return self.post_tokenize(intermediate_text)

    def __len__(self):
        return len(self.base_tokenizer)
