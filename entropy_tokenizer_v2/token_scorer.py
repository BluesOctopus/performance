"""Stage 3: Score(w)=ΔT/(I(w)+ε); map high-score tokens to <VAR>/<STR>/… placeholders."""

import builtins
import io
import keyword
import math
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from config import (
    PLACEHOLDERS,
    SCORE_EPSILON,
    SCORE_THRESHOLD_PERCENTILE,
    VOCAB_COST_MODE,
)

_KEYWORDS  = set(keyword.kwlist)
_BUILTINS  = set(dir(builtins))
_PROTECTED = _KEYWORDS | _BUILTINS | {
    "self", "cls", "__init__", "__name__", "__main__",
    "True", "False", "None", "args", "kwargs",
}


def _safe_tokenize(source: str) -> list:
    try:
        return list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return []


def _extract_vocab_from_source(source: str) -> dict[str, Counter]:
    """Per-category token counts from ``tokenize``."""
    toks = _safe_tokenize(source)

    ids:    Counter = Counter()
    attrs:  Counter = Counter()
    strs:   Counter = Counter()
    fstrs:  Counter = Counter()
    nums:   Counter = Counter()

    prev_is_dot = False

    for tok in toks:
        ttype, tstr = tok.type, tok.string

        if ttype == tokenize.NAME:
            if tstr not in _PROTECTED:
                ids[tstr] += 1
                if prev_is_dot:
                    attrs[tstr] += 1
            prev_is_dot = False

        elif ttype == tokenize.STRING:
            if tstr.startswith(("f'", 'f"', "f'''", 'f"""',
                                 "F'", 'F"', "F'''", 'F"""',
                                 "rf'", 'rf"', "fr'", 'fr"')):
                fstrs[tstr] += 1
            else:
                strs[tstr] += 1
            prev_is_dot = False

        elif ttype == tokenize.NUMBER:
            nums[tstr] += 1
            prev_is_dot = False

        elif ttype == tokenize.OP:
            prev_is_dot = (tstr == ".")

        else:
            prev_is_dot = False

    return {
        "identifiers": ids,
        "attributes":  attrs,
        "strings":     strs,
        "fstrings":    fstrs,
        "numbers":     nums,
    }


def build_vocabulary(sources: list[str]) -> dict[str, Counter]:
    """Merge per-file vocab into corpus-wide Counters."""
    total: dict[str, Counter] = {
        "identifiers": Counter(),
        "attributes":  Counter(),
        "strings":     Counter(),
        "fstrings":    Counter(),
        "numbers":     Counter(),
    }
    for src in sources:
        for cat, counter in _extract_vocab_from_source(src).items():
            total[cat].update(counter)
    return total


@dataclass
class TokenInfo:
    word:      str
    category:  str
    freq:      int
    spt:       float    # subtoken count under target tokenizer
    self_info: float    # I(w) = -log2 p(w)
    delta_T:   float    # (spt - 1) * freq
    score:     float    # delta_T / (self_info + ε)


def _spt_estimate(word: str, tokenizer=None, tok_type: str = "tiktoken") -> float:
    if tokenizer is None:
        return max(1.0, len(word) / 3.5)
    try:
        if tok_type == "tiktoken":
            return float(len(tokenizer.encode(word, allowed_special="all")))
        return float(len(tokenizer.encode(word, add_special_tokens=False)))
    except Exception:
        return max(1.0, len(word) / 3.5)


def compute_scores(
    vocab: dict[str, Counter],
    tokenizer=None,
    tok_type: str = "tiktoken",
    epsilon: float = SCORE_EPSILON,
) -> dict[str, TokenInfo]:
    """p(w) pooled over all categories."""
    ids   = vocab["identifiers"]
    attrs = vocab["attributes"]
    strs  = vocab["strings"]
    fstrs = vocab["fstrings"]
    nums  = vocab["numbers"]

    total_freq = (sum(ids.values()) + sum(strs.values())
                  + sum(fstrs.values()) + sum(nums.values()))
    if total_freq == 0:
        return {}

    results: dict[str, TokenInfo] = {}

    def _add(word: str, freq: int, cat: str):
        if freq == 0:
            return
        p_w       = freq / total_freq
        self_info = -math.log2(p_w)
        spt       = _spt_estimate(word, tokenizer, tok_type)
        delta_T   = max(0.0, spt - 1.0) * freq
        score     = delta_T / (self_info + epsilon)
        results[word] = TokenInfo(
            word=word, category=cat, freq=freq,
            spt=spt, self_info=self_info,
            delta_T=delta_T, score=score,
        )

    for word, freq in ids.items():
        cat = "attribute" if word in attrs else "variable"
        _add(word, freq, cat)

    for word, freq in strs.items():
        _add(word, freq, "string")

    for word, freq in fstrs.items():
        _add(word, freq, "fstring")

    for word, freq in nums.items():
        _add(word, freq, "number")

    return results


def select_replacement_set(
    scores: dict[str, TokenInfo],
    threshold_percentile: float = SCORE_THRESHOLD_PERCENTILE,
) -> set[str]:
    """Tokens with spt>1 and score ≥ quantile cutoff."""
    eligible = [info for info in scores.values() if info.spt > 1.0]
    if not eligible:
        return set()

    eligible.sort(key=lambda x: x.score)
    cutoff_idx = int(len(eligible) * threshold_percentile)
    cutoff_score = eligible[cutoff_idx].score if cutoff_idx < len(eligible) else float("inf")

    return {info.word for info in eligible if info.score >= cutoff_score}


def build_replacement_map(
    scores: dict[str, TokenInfo],
    replacement_set: set[str],
) -> dict[str, str]:
    rmap: dict[str, str] = {}
    for word in replacement_set:
        if word in scores:
            placeholder = PLACEHOLDERS.get(scores[word].category, "<UNK>")
            rmap[word] = placeholder
    return rmap


_TOKENIZE_IGNORED_TYPES = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
}


def _line_start_offsets(text: str) -> list[int]:
    offsets: list[int] = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def _pos_to_offset(line_starts: list[int], pos: tuple[int, int]) -> int:
    line_no, col = pos
    return line_starts[line_no - 1] + col


def _has_overlap(start: int, end: int, protected_spans: list[tuple[int, int]]) -> bool:
    for p_start, p_end in protected_spans:
        if start < p_end and p_start < end:
            return True
    return False


def _iter_safe_token_replacements(
    text: str,
    rmap: dict[str, str],
    protected_spans: Optional[list[tuple[int, int]]] = None,
) -> list[tuple[int, int, str]]:
    if not text or not rmap:
        return []

    line_starts = _line_start_offsets(text)
    replacements: list[tuple[int, int, str]] = []

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []

    prev_sig = None
    protected_spans = protected_spans or []
    for tok in tokens:
        ttype = tok.type
        tstr = tok.string
        repl: Optional[str] = None

        if ttype == tokenize.STRING:
            repl = rmap.get(tstr)
            if repl not in (PLACEHOLDERS["string"], PLACEHOLDERS["fstring"]):
                repl = None
        elif ttype == tokenize.NUMBER:
            repl = rmap.get(tstr)
            if repl != PLACEHOLDERS["number"]:
                repl = None
        elif ttype == tokenize.NAME:
            repl = rmap.get(tstr)
            if repl is not None:
                is_attr = (
                    prev_sig is not None
                    and prev_sig.type == tokenize.OP
                    and prev_sig.string == "."
                )
                if is_attr and repl != PLACEHOLDERS["attribute"]:
                    repl = None
                if (not is_attr) and repl != PLACEHOLDERS["variable"]:
                    repl = None

        if repl is not None:
            start = _pos_to_offset(line_starts, tok.start)
            end = _pos_to_offset(line_starts, tok.end)
            if not _has_overlap(start, end, protected_spans):
                replacements.append((start, end, repl))

        if ttype not in _TOKENIZE_IGNORED_TYPES and ttype != tokenize.ENDMARKER:
            prev_sig = tok

    return replacements


def apply_token_replacement(text: str, rmap: dict[str, str]) -> str:
    """Token-aware replacement; only exact full-token matches are replaced."""
    return apply_token_replacement_with_protected_spans(text, rmap, [])


def apply_token_replacement_with_protected_spans(
    text: str,
    rmap: dict[str, str],
    protected_spans: list[tuple[int, int]],
) -> str:
    """Token-aware replacement while skipping protected character spans."""
    if not rmap:
        return text

    spans = _iter_safe_token_replacements(text, rmap, protected_spans=protected_spans)
    if not spans:
        return text

    out = text
    for start, end, repl in reversed(spans):
        out = out[:start] + repl + out[end:]
    return out


STAGE3_VOCAB_DEFINITIONS: dict[str, str] = {
    PLACEHOLDERS["variable"]: "identifier placeholder",
    PLACEHOLDERS["attribute"]: "attribute placeholder",
    PLACEHOLDERS["string"]: "string placeholder",
    PLACEHOLDERS["fstring"]: "fstring placeholder",
    PLACEHOLDERS["number"]: "number placeholder",
}


def build_stage3_vocab_entries_from_replacements(
    replacement_map: dict[str, str],
) -> list[dict]:
    """One entry per distinct placeholder token that appears as a replacement value."""
    seen: set[str] = set()
    out: list[dict] = []
    for ph in sorted(set(replacement_map.values())):
        if ph in STAGE3_VOCAB_DEFINITIONS and ph not in seen:
            seen.add(ph)
            out.append(
                {
                    "token": ph,
                    "kind": "stage3",
                    "definition": STAGE3_VOCAB_DEFINITIONS[ph],
                }
            )
    return out


def build_stage3_vocab_entries_from_used_placeholders(
    used_placeholders: list[str],
) -> list[dict]:
    """Entries for placeholders actually present (caller supplies stable order)."""
    out: list[dict] = []
    seen: set[str] = set()
    for ph in used_placeholders:
        if ph in STAGE3_VOCAB_DEFINITIONS and ph not in seen:
            seen.add(ph)
            out.append(
                {
                    "token": ph,
                    "kind": "stage3",
                    "definition": STAGE3_VOCAB_DEFINITIONS[ph],
                }
            )
    return out


def collect_used_stage3_placeholders(text: str, replacement_map: dict[str, str]) -> list[str]:
    """Placeholders from *text* that are values in *replacement_map*, first-seen order."""
    from placeholder_accounting import extract_placeholders

    allowed = set(replacement_map.values())
    seen: set[str] = set()
    out: list[str] = []
    for ph in extract_placeholders(text):
        if ph in allowed and ph not in seen:
            seen.add(ph)
            out.append(ph)
    return out


def estimate_stage3_replacement_sequence_gain(
    original_text: str,
    replacement_text: str,
    *,
    tokenizer=None,
    tok_type: Optional[str] = None,
) -> dict:
    from placeholder_accounting import count_sequence_tokens

    b = count_sequence_tokens(original_text, tokenizer=tokenizer, tok_type=tok_type)
    r = count_sequence_tokens(replacement_text, tokenizer=tokenizer, tok_type=tok_type)
    return {
        "baseline_sequence_tokens": b,
        "replacement_sequence_tokens": r,
        "sequence_net_saving": b - r,
    }


def estimate_stage3_effective_gain(
    original_text: str,
    replaced_text: str,
    used_placeholders: list[str],
    *,
    vocab_cost_mode: str = VOCAB_COST_MODE,
    tokenizer=None,
    tok_type: Optional[str] = None,
) -> dict:
    from placeholder_accounting import (
        compute_vocab_intro_cost,
        count_sequence_tokens,
    )

    total_baseline_sequence_tokens = count_sequence_tokens(
        original_text, tokenizer=tokenizer, tok_type=tok_type
    )
    total_replacement_sequence_tokens = count_sequence_tokens(
        replaced_text, tokenizer=tokenizer, tok_type=tok_type
    )
    total_sequence_net_saving = (
        total_baseline_sequence_tokens - total_replacement_sequence_tokens
    )
    vocab_entries = build_stage3_vocab_entries_from_used_placeholders(used_placeholders)
    vocab_intro_tokens = compute_vocab_intro_cost(
        vocab_entries,
        mode=vocab_cost_mode,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )
    effective_total_net_saving = total_sequence_net_saving - vocab_intro_tokens
    return {
        "total_baseline_sequence_tokens": total_baseline_sequence_tokens,
        "total_replacement_sequence_tokens": total_replacement_sequence_tokens,
        "total_sequence_net_saving": total_sequence_net_saving,
        "vocab_intro_tokens": vocab_intro_tokens,
        "effective_total_net_saving": effective_total_net_saving,
    }


def context_aware_window_token_delta(
    full_text: str,
    span_start: int,
    span_end: int,
    literal_slice: str,
    replacement_slice: str,
    tokenizer,
    tok_type: str,
    *,
    window_chars: int = 80,
) -> int:
    """
    Approximate tokenizer delta for one occurrence inside a fixed char window.

    Returns (tokens with literal) - (tokens with replacement); positive means the
    replacement saves sequence tokens in that window. This captures local merge
    effects better than per-token raw_cost - alias_cost.
    """
    from marker_count import encode as mc_encode

    w0 = max(0, span_start - int(window_chars))
    w1 = min(len(full_text), span_end + int(window_chars))
    frag = full_text[w0:w1]
    ls = span_start - w0
    le = span_end - w0
    if frag[ls:le] != literal_slice:
        # Defensive: window mis-aligned (should not happen for token spans).
        return 0
    with_lit = frag
    with_alias = frag[:ls] + replacement_slice + frag[le:]
    return len(mc_encode(tokenizer, tok_type, with_lit)) - len(
        mc_encode(tokenizer, tok_type, with_alias)
    )


def sum_context_aware_literal_delta(
    full_text: str,
    spans: list[tuple[int, int]],
    literal_slice: str,
    replacement_slice: str,
    tokenizer,
    tok_type: str,
    *,
    window_chars: int = 80,
) -> int:
    """Sum per-occurrence window deltas for the same literal→alias rewrite."""
    total = 0
    for st, ed in spans:
        total += context_aware_window_token_delta(
            full_text,
            st,
            ed,
            literal_slice,
            replacement_slice,
            tokenizer,
            tok_type,
            window_chars=window_chars,
        )
    return total


def score_summary(scores: dict[str, TokenInfo], top_n: int = 20) -> list[dict]:
    """Top-*n* by score as dict rows."""
    top = sorted(scores.values(), key=lambda x: x.score, reverse=True)[:top_n]
    return [
        {
            "word":      t.word,
            "category":  t.category,
            "score":     round(t.score, 4),
            "delta_T":   round(t.delta_T, 1),
            "self_info": round(t.self_info, 4),
            "freq":      t.freq,
            "spt":       round(t.spt, 2),
        }
        for t in top
    ]
