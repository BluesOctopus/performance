from __future__ import annotations

from config import EVAL_TOKENIZERS
from repo_miner import _load_tokenizer
from stage3.lexical.semantic_codec import encode_semantic_strings


def test_semantic_codec_clusters_similar_texts():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    text = (
        "a = 'Please verify the user login request before proceeding now and include account context details for the current session'\n"
        "b = 'Please verify user login request before proceeding immediately and include account context details for current session'\n"
        "c = 'Please verify the user login request before proceeding now and include account context details for the current session'\n"
        "d = 'Please verify user login request before proceeding immediately and include account context details for current session'\n"
    )
    res = encode_semantic_strings(
        text,
        tokenizer=tok,
        tok_type=tt,
        similarity_threshold=0.70,
        risk_threshold=0.60,
        min_cluster_size=2,
    )
    assert res.candidates >= 2
    assert res.used_clusters >= 1
    assert res.similarity_kind == "lexical_bow_cosine"
    assert res.mode == "lexical_free_text_baseline"


def test_semantic_codec_low_similarity_fallback():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    text = (
        "a = 'database connection timeout happened in worker'\n"
        "b = 'rendering colorful chart with bar values'\n"
    )
    res = encode_semantic_strings(
        text,
        tokenizer=tok,
        tok_type=tt,
        similarity_threshold=0.95,
        risk_threshold=0.95,
        min_cluster_size=2,
    )
    assert res.used_clusters == 0
