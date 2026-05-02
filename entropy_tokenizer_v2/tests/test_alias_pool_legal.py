"""Legal identifier alias pool (hybrid_ab A channel)."""

from __future__ import annotations

import keyword

import pytest

from config import EVAL_TOKENIZERS
from repo_miner import _load_tokenizer
from stage3.literal_codec.alias_pool import build_legal_alias_alphabet, is_legal_public_identifier


def test_is_legal_rejects_keywords() -> None:
    assert is_legal_public_identifier("for") is False
    assert is_legal_public_identifier("valid_x") is True


def test_build_legal_alias_alphabet_respects_reserved() -> None:
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    reserved = set(keyword.kwlist) | {"myfn"}
    pool = build_legal_alias_alphabet(
        tok, tt, reserved=reserved, max_n=64, max_alias_token_len=2
    )
    assert pool
    for name in pool:
        assert name not in reserved
        assert name not in keyword.kwlist
        assert is_legal_public_identifier(name)
