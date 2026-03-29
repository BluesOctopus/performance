"""Stage 3: Score(w)=ΔT/(I(w)+ε); map high-score tokens to <VAR>/<STR>/… placeholders."""

import builtins
import io
import keyword
import math
import re
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from config import SCORE_EPSILON, SCORE_THRESHOLD_PERCENTILE, PLACEHOLDERS

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


_ENCODE_CACHE = {}

def build_local_replacement_map(
    text: str,
    tokenizer,
    tok_type: str,
    min_saving: int = 1
) -> tuple[dict[str, str], str]:
    """
    Build a per-file local replacement map for identifiers.
    Returns (replacement_map, registry_header).
    """
    from marker_count import encode as mc_encode
    import re
    from collections import Counter

    # 1. Extract all potential identifiers (variables and attributes)
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
    counts = Counter(words)
    
    # Filter out Python keywords and special markers
    import keyword
    marker_pattern = re.compile(r'^SYN_\d+$|^<SYN_\d+>$|^<D>$|^<I>$|^<VAR.*>$|^<ATTR.*>$|^<STR.*>$|^<FSTR.*>$|^<NUM.*>$|^<V\d+>$|^<R>$|^</R>$')
    candidates = {w: freq for w, freq in counts.items() 
                  if not keyword.iskeyword(w) 
                  and len(w) > 2 
                  and not marker_pattern.match(w)}
    
    if not candidates:
        return {}, ""

    # 2. Calculate savings for each candidate
    scored_candidates = []
    for word, freq in candidates.items():
        if word not in _ENCODE_CACHE:
            _ENCODE_CACHE[word] = len(mc_encode(tokenizer, tok_type, word))
        orig_toks = _ENCODE_CACHE[word]
        
        overhead = orig_toks + 1
        net_saved = (orig_toks - 1) * freq - overhead
        if net_saved >= min_saving:
            scored_candidates.append((word, net_saved))
            
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    if not scored_candidates:
        return {}, ""

    # 3. Build map and header
    rmap = {}
    registry_words = []
    for i, (word, _) in enumerate(scored_candidates):
        rmap[word] = f"<V{i}>"
        registry_words.append(word)
        
    header = f"<R>{','.join(registry_words)}</R>"
    return rmap, header


def apply_local_token_replacement(text: str, rmap: dict[str, str], header: str) -> str:
    """Apply local replacements and prepend header."""
    if not rmap:
        return text
        
    import re
    def replace_func(match):
        word = match.group(0)
        return rmap.get(word, word)
    
    pattern = re.compile(r'\b(' + '|'.join(re.escape(w) for w in sorted(rmap.keys(), key=len, reverse=True)) + r')\b')
    compressed_text = pattern.sub(replace_func, text)
    
    return f"{header}\n{compressed_text}"


def build_replacement_map(
    scores: dict[str, TokenInfo],
    replacement_set: set[str],
) -> dict[str, str]:
    """Build a stable replacement map with unique indices per category."""
    rmap: dict[str, str] = {}
    
    sorted_words = sorted(
        [scores[w] for w in replacement_set if w in scores],
        key=lambda x: (-x.score, x.word)
    )
    
    counters = {
        "variable": 0,
        "attribute": 0,
        "string": 0,
        "fstring": 0,
        "number": 0
    }
    
    for info in sorted_words:
        cat = info.category
        idx = counters.get(cat, 0)
        prefix = PLACEHOLDERS.get(cat, "<UNK>")
        rmap[info.word] = f"{prefix[:-1]}_{idx}>"
        counters[cat] += 1
        
    return rmap


_IDENT_RE  = re.compile(r'\b([A-Za-z_]\w*)\b')
_STRING_RE = re.compile(
    r'(?:f|F|r|R|b|B|u|U)?(?:\'\'\'[\s\S]*?\'\'\'|"""[\s\S]*?"""|'
    r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"
)
_NUMBER_RE = re.compile(r'\b(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[jJ]?|\d+)\b')


def apply_token_replacement(text: str, rmap: dict[str, str]) -> str:
    """Replace strings, then numbers, then identifiers using the stable mapping."""
    if not rmap:
        return text

    str_words = {w: p for w, p in rmap.items() if "<STR_" in p or "<FSTR_" in p}
    num_words = {w: p for w, p in rmap.items() if "<NUM_" in p}
    id_words  = {w: p for w, p in rmap.items() if "<VAR_" in p or "<ATTR_" in p}

    if str_words:
        def _replace_str(m):
            s = m.group(0)
            return str_words.get(s, s)
        text = _STRING_RE.sub(_replace_str, text)

    if num_words:
        def _replace_num(m):
            n = m.group(0)
            return num_words.get(n, n)
        text = _NUMBER_RE.sub(_replace_num, text)

    if id_words:
        def _replace_id(m):
            word = m.group(1)
            return id_words.get(word, word)
        text = _IDENT_RE.sub(_replace_id, text)

    return text


def reverse_local_token_replacement(compressed_text: str) -> str:
    """Extract registry header and restore identifiers."""
    match = re.match(r'^<R>(.*?)</R>\n', compressed_text)
    if not match:
        return compressed_text
        
    registry_str = match.group(1)
    body = compressed_text[match.end():]
    
    if not registry_str:
        return body
        
    words = registry_str.split(',')
    rmap_rev = {f"<V{i}>": word for i, word in enumerate(words)}
    
    def replace_func(match):
        marker = match.group(0)
        return rmap_rev.get(marker, marker)
        
    pattern = re.compile(r'<V\d+>')
    return pattern.sub(replace_func, body)

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
