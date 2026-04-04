"""File-level Stage3 hybrid_ab guardrail (GPT-4 profile)."""

from __future__ import annotations

import pytest

from stage3.backends.hybrid_ab_backend import (
    HybridABConfig,
    HybridABResult,
    _apply_hybrid_ab_file_guardrail,
)
from stage3.exact.alias_codec import ACodecResult, AEntry
from stage3.lexical.semantic_codec import BCodecResult


def test_guardrail_full_fallback_when_sequence_tokens_worse(monkeypatch: pytest.MonkeyPatch) -> None:
    from stage3.backends import hybrid_ab_backend as hab

    s2 = "aaa"
    occ = {("variable", "longname"): [(0, 8)]}
    entries = [
        AEntry(
            field="variable",
            literal="longname",
            alias="x",
            count=1,
            raw_cost=3,
            alias_cost=1,
            intro_cost=1,
            gain=1,
        )
    ]
    a_res = ACodecResult(
        encoded_text="x" * 8,
        entries=entries,
        candidates=1,
        selected=1,
        used_entries=1,
        occ=occ,
    )
    b_res = BCodecResult(encoded_text="inflated_final_text")
    raw = HybridABResult(encoded_text=b_res.encoded_text, a=a_res, b=b_res, fallback_count=0, meta={})

    def _fake_len(text: str, _tok, _tt) -> int:
        if text == s2:
            return 10
        if text == b_res.encoded_text:
            return 50
        return len(text)

    monkeypatch.setattr(hab, "_sequence_token_len", _fake_len)
    conf = HybridABConfig(enable_global_guardrail=True, enable_incremental_rollback=False)
    final, telem = _apply_hybrid_ab_file_guardrail(s2, raw, conf=conf, tokenizer=object(), tok_type="tiktoken")
    assert final.encoded_text == s2
    assert telem["stage3_guardrail_triggered"] is True
    assert telem["num_candidates_applied"] == 0


def test_guardrail_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from stage3.backends import hybrid_ab_backend as hab

    s2 = "x = 1\n"
    a_res = ACodecResult(encoded_text=s2)
    b_res = BCodecResult(encoded_text=s2)
    raw = HybridABResult(encoded_text=s2, a=a_res, b=b_res, fallback_count=0, meta={})
    conf = HybridABConfig(enable_global_guardrail=False)
    monkeypatch.setattr(hab, "_sequence_token_len", lambda *_a, **_k: 5)
    final, telem = _apply_hybrid_ab_file_guardrail(
        s2, raw, conf=conf, tokenizer=object(), tok_type="tiktoken"
    )
    assert final.encoded_text == s2
    assert telem["stage3_guardrail_triggered"] is False
