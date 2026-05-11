from config import EVAL_TOKENIZERS
from repo_miner import _load_tokenizer
from marker_count import encode as mc_encode
from stage3.exact.alias_codec import (
    build_alias_alphabet,
    decode_exact_aliases,
    encode_exact_aliases,
)
from stage3.routing.router import ABRoutingConfig


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


def test_alias_codec_protects_top_level_symbols():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    text = (
        "def public_api_function():\n"
        "    return 1\n"
        "class ExportedClass:\n"
        "    pass\n"
        "__all__ = ['public_api_function', 'ExportedClass']\n"
        "public_api_function()\n"
        "public_api_function()\n"
    )
    res = encode_exact_aliases(
        text,
        tokenizer=tok,
        tok_type=tt,
        route_cfg=ABRoutingConfig(),
    )
    assert "public_api_function" in res.encoded_text
    assert "ExportedClass" in res.encoded_text


def test_alias_codec_private_helper_can_be_aliased():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    text = (
        "def _private_helper():\n"
        "    return 1\n"
        "\n"
        + "\n".join([f"x{i} = _private_helper()" for i in range(10)])
        + "\n"
    )
    res = encode_exact_aliases(
        text,
        tokenizer=tok,
        tok_type=tt,
        route_cfg=ABRoutingConfig(),
        min_net_gain=-10**9,
    )
    assert any(e.literal == "_private_helper" for e in res.entries)
    # Alias alphabet is tokenizer-sorted (often 1-token names like ``x0``), not ``__ab*``.
    assert "_private_helper" not in res.encoded_text
    assert res.encoded_text.startswith("def ")


def test_alias_codec_mnemonic_style_prefix():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    text = (
        "verylongsymbol_for_alias = 1\n"
        "print(verylongsymbol_for_alias)\n"
        "verylongsymbol_for_alias += 1\n"
    )
    res = encode_exact_aliases(
        text,
        tokenizer=tok,
        tok_type=tt,
        route_cfg=ABRoutingConfig(),
        alias_style="mnemonic",
    )
    if res.entries:
        assert res.entries[0].alias.startswith("_ve")


def test_alias_alphabet_sorted_by_token_cost():
    tok, tt = _load_tokenizer("gpt4", EVAL_TOKENIZERS["gpt4"])
    cands = build_alias_alphabet(tok, tt, style="short", max_n=16)
    costs = [len(mc_encode(tok, tt, c)) for c in cands]
    assert costs == sorted(costs)
