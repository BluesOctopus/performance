"""Unified placeholder accounting (sequence + vocab intro)."""

import json

from config import FIXED_VOCAB_TOKEN_COST
from markers import is_placeholder_token
from placeholder_accounting import (
    compute_effective_total_tokens,
    compute_vocab_intro_cost,
    count_base_tokens,
    count_sequence_tokens,
    serialize_vocab_entry,
    split_text_by_placeholders,
)
from syntax_compressor import (
    build_stage1_vocab_entry,
    estimate_stage1_candidate_effective_gain,
)
from token_scorer import estimate_stage3_effective_gain


class SplitTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False, allowed_special: str = "all"):
        del add_special_tokens, allowed_special
        return text.split()


class CharTokenizer:
    """One pseudo-token per character (exaggerates raw text vs short placeholder lines)."""

    def encode(self, text: str, add_special_tokens: bool = False, allowed_special: str = "all"):
        del add_special_tokens, allowed_special
        return list(text)


def test_syn_and_var_count_as_one_sequence_token() -> None:
    tok = SplitTokenizer()
    assert count_sequence_tokens("<SYN_0>", tokenizer=tok, tok_type="hf") == 1
    assert count_sequence_tokens("<VAR>", tokenizer=tok, tok_type="hf") == 1


def test_placeholder_not_split_inside_larger_text() -> None:
    tok = SplitTokenizer()
    t = "foo <SYN_0> bar"
    assert count_sequence_tokens(t, tokenizer=tok, tok_type="hf") == 3  # foo, <SYN_0>, bar


def test_split_text_by_placeholders_structure() -> None:
    assert split_text_by_placeholders("a<VAR>b") == [("a", False), ("<VAR>", True), ("b", False)]


def test_vocab_intro_fixed_per_token_two_entries() -> None:
    entries = [
        {"token": "<SYN_0>", "kind": "stage1", "definition": "x"},
        {"token": "<SYN_1>", "kind": "stage1", "definition": "y"},
    ]
    c = compute_vocab_intro_cost(entries, mode="fixed_per_token", tokenizer=None, tok_type=None)
    assert c == 2 * FIXED_VOCAB_TOKEN_COST


def test_vocab_intro_serialized_uses_base_tokens_not_placeholder_rules() -> None:
    tok = SplitTokenizer()
    entry = {"token": "<SYN_2>", "kind": "stage1", "definition": "return {0}"}
    s = serialize_vocab_entry(entry)
    cost_vocab = compute_vocab_intro_cost(
        [entry], mode="serialized_definition", tokenizer=tok, tok_type="hf"
    )
    cost_direct = count_base_tokens(s, tokenizer=tok, tok_type="hf")
    assert cost_vocab == cost_direct
    # Serialized string is counted as raw text, not placeholder-aware.
    naive_ph = count_sequence_tokens(s, tokenizer=tok, tok_type="hf")
    assert "<SYN_2>" in s
    assert isinstance(naive_ph, int)


def test_compute_effective_total_tokens_sum() -> None:
    tok = SplitTokenizer()
    text = "a b"
    entries = [{"token": "<VAR>", "kind": "stage3", "definition": "id"}]
    d = compute_effective_total_tokens(
        text,
        entries,
        vocab_cost_mode="fixed_per_token",
        tokenizer=tok,
        tok_type="hf",
    )
    assert d["sequence_only_tokens"] == 2
    assert d["vocab_intro_tokens"] == FIXED_VOCAB_TOKEN_COST
    assert d["effective_total_tokens"] == 2 + FIXED_VOCAB_TOKEN_COST


def test_stage1_effective_negative_when_vocab_intro_large(monkeypatch) -> None:
    tok = CharTokenizer()
    occ = [("    return x\n", ["x"]), ("    return y\n", ["y"])]
    monkeypatch.setattr("syntax_compressor.compute_vocab_intro_cost", lambda *_a, **_k: 9999)
    eg = estimate_stage1_candidate_effective_gain(
        "return {0}",
        occ,
        "<SYN_0>",
        tokenizer=tok,
        tok_type="hf",
    )
    assert eg["total_sequence_net_saving"] > 0
    assert eg["effective_total_net_saving"] < 0


def test_stage3_effective_negative_when_vocab_dominates() -> None:
    tok = CharTokenizer()
    original = "hello"
    replaced = "hi"
    used = ["<VAR>", "<ATTR>", "<STR>", "<NUM>"]
    g = estimate_stage3_effective_gain(
        original,
        replaced,
        used,
        vocab_cost_mode="fixed_per_token",
        tokenizer=tok,
        tok_type="hf",
    )
    seq = g["total_sequence_net_saving"]
    assert seq > 0
    assert g["vocab_intro_tokens"] == len(used) * FIXED_VOCAB_TOKEN_COST
    assert g["effective_total_net_saving"] == seq - g["vocab_intro_tokens"]


def test_is_placeholder_token() -> None:
    assert is_placeholder_token("<SYN_0>")
    assert is_placeholder_token("<VAR>")
    assert not is_placeholder_token("VAR")


def test_stage1_vocab_json_stable(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    j1 = tmp_path / "v1.json"
    j2 = tmp_path / "v2.json"
    cmd = [
        sys.executable,
        "scripts/run_ablation.py",
        "--summary-output",
        str(tmp_path / "s.csv"),
        "--per-file-output",
        str(tmp_path / "p.csv"),
        "--stage1-selected-output",
        str(tmp_path / "c.csv"),
        "--stage1-vocab-json",
        str(j1),
        "--stage3-vocab-json",
        str(tmp_path / "t3.json"),
    ]
    subprocess.run(cmd, cwd=root, check=True, capture_output=True)
    subprocess.run(
        cmd[:-2] + ["--stage1-vocab-json", str(j2), "--stage3-vocab-json", str(tmp_path / "t3b.json")],
        cwd=root,
        check=True,
        capture_output=True,
    )
    assert json.loads(j1.read_text(encoding="utf-8")) == json.loads(
        j2.read_text(encoding="utf-8")
    )
