from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_S3 = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_S3) not in sys.path:
    sys.path.insert(0, str(_S3))

from config import EVAL_TOKENIZERS
from repo_miner import _load_tokenizer
from aliasing.alias_codec import decode_exact_aliases, encode_exact_aliases
from router import ABRoutingConfig


def test_alias_codec_selects_repeated_and_roundtrip():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    text = (
        "very_long_variable_name_for_alias_test = 1\n"
        "print(very_long_variable_name_for_alias_test)\n"
        "very_long_variable_name_for_alias_test += 1\n"
        "print(very_long_variable_name_for_alias_test)\n"
    )
    res = encode_exact_aliases(
        text,
        tokenizer=tok,
        tok_type=tt,
        route_cfg=ABRoutingConfig(),
    )
    assert res.candidates >= 1
    assert res.selected >= 1
    back = decode_exact_aliases(res.encoded_text, res.entries)
    assert back == text


def test_alias_codec_single_occurrence_not_selected():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    text = "single_token_only_once = 1\n"
    res = encode_exact_aliases(
        text,
        tokenizer=tok,
        tok_type=tt,
        route_cfg=ABRoutingConfig(),
    )
    assert res.selected == 0
