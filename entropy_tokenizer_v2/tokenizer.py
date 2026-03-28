"""Neural-Symbolic Tokenizer Implementation.
Integrates Stage 1 (Skeleton), Stage 2 (Cleaning), and Stage 3 (Local Registry)
into a unified interface for code compression and decompression.
"""

import re
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass

from repo_miner import RepoConfig, _load_tokenizer, _vocab_size
from syntax_compressor import compress_source_syntax, decompress_source_syntax
from lossy_cleaner import CleaningConfig, clean_code, restore_code
from token_scorer import build_local_replacement_map, apply_local_token_replacement, reverse_local_token_replacement
from marker_count import count_augmented, RE_ALL_MARKERS, encode as _encode_with_markers

@dataclass
class TokenizerOutput:
    text: str
    token_count: int
    metadata: Dict[str, Any]

class NeuralSymbolicTokenizer:
    def __init__(
        self,
        tokenizer_key: str = "gpt4",
        repo_config: Optional[RepoConfig] = None,
        cleaning_config: Optional[CleaningConfig] = None
    ):
        """
        Initialize the Neural-Symbolic Tokenizer.
        
        Args:
            tokenizer_key: The base LLM tokenizer to use (e.g., 'gpt4', 'gpt2').
            repo_config: Pre-mined global configuration (Stage 1 skeletons).
            cleaning_config: Configuration for Stage 2 cleaning.
        """
        from config import EVAL_TOKENIZERS
        self.tokenizer_key = tokenizer_key
        self.tokenizer_cfg = EVAL_TOKENIZERS.get(tokenizer_key)
        if not self.tokenizer_cfg:
            raise ValueError(f"Unknown tokenizer key: {tokenizer_key}")
            
        self.base_tokenizer, self.tok_type = _load_tokenizer(tokenizer_key, self.tokenizer_cfg)
        self.repo_config = repo_config
        
        if cleaning_config is None:
            self.cleaning_config = CleaningConfig(
                remove_comments=False,
                remove_blank_lines=False, # Keep blank lines for AST exact match
                remove_trailing_whitespace=True,
                remove_docstrings=False,
                use_minimalist_indent=True,
            )
        else:
            self.cleaning_config = cleaning_config

    def encode(self, text: str, add_special_tokens: bool = True) -> TokenizerOutput:
        """
        Compress code using the 3-stage pipeline.
        """
        # 1. Stage 2: Cleaning & Structural Indent (Lossless Semantic)
        # We run this FIRST so that Stage 1 sees the code without physical indents
        # but with colons still present.
        compressed_s2, _ = clean_code(text, self.cleaning_config)
        if compressed_s2 is None: compressed_s2 = text
            
        # 2. Stage 1: Syntax Skeleton (Global)
        if self.repo_config:
            skeletons = self.repo_config.skeleton_candidates()
            compressed_s1 = compress_source_syntax(compressed_s2, skeletons)
        else:
            compressed_s1 = compressed_s2
        if compressed_s1 is None: compressed_s1 = compressed_s2
        
        # 3. Stage 3: Local Identifier Replacement
        lmap, lheader = build_local_replacement_map(compressed_s1, self.base_tokenizer, self.tok_type)
        compressed_s3 = apply_local_token_replacement(compressed_s1, lmap, lheader)
        if compressed_s3 is None: compressed_s3 = compressed_s1
        
        final_text = compressed_s3
        token_count = count_augmented(final_text, self.base_tokenizer, self.tok_type, pattern=RE_ALL_MARKERS)
        
        return TokenizerOutput(
            text=final_text,
            token_count=token_count,
            metadata={
                "local_map": lmap,
                "local_header": lheader,
                "stage1_applied": self.repo_config is not None
            }
        )

    def decode(self, compressed_text: str) -> str:
        """
        Decompress the text back to semantically equivalent code.
        """
        # 1. Reverse Stage 3: Local Identifiers
        decompressed_s3 = reverse_local_token_replacement(compressed_text)
        
        # 2. Reverse Stage 1: Syntax Skeletons
        if self.repo_config:
            skeletons = self.repo_config.skeleton_candidates()
            decompressed_s1 = decompress_source_syntax(decompressed_s3, skeletons)
        else:
            decompressed_s1 = decompressed_s3

        # 3. Reverse Stage 2: Structural Indents
        decompressed_s2 = restore_code(decompressed_s1, self.cleaning_config)
        
        return decompressed_s2

    def tokenize(self, text: str) -> List[Union[int, str]]:
        """
        A helper that returns a list of mixed token IDs (from base tokenizer) 
        and special marker strings.
        """
        compressed = self.encode(text).text
        return _encode_with_markers(self.base_tokenizer, self.tok_type, compressed)

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        HF-style __call__ interface.
        """
        output = self.encode(text)
        tokens = self.tokenize(text)
        return {
            "input_ids": tokens,
            "compressed_text": output.text,
            "token_count": output.token_count,
            "metadata": output.metadata
        }
