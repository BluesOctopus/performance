"""Experimental reversible newline+indent layout tokens (<NL0>…<NL8>).

Conservative: skips ambiguous lines, multiline string bodies, and tokenize failures.
Never writes to disk paths here — callers own persistence.
"""

from __future__ import annotations

import io
import re
import tokenize
from math import gcd
from functools import reduce
from typing import Any

from config import VOCAB_COST_MODE
from lossy_cleaner import find_multiline_string_line_spans
from placeholder_accounting import compute_vocab_intro_cost, count_sequence_tokens

LAYOUT_TOKEN_MAX_DEPTH = 8
LAYOUT_TOKEN_RE = re.compile(r"^<NL([0-8])>$")
LAYOUT_TOKEN_FIND_RE = re.compile(r"<NL([0-8])>")


def build_layout_token(depth: int) -> str:
    if depth < 0 or depth > LAYOUT_TOKEN_MAX_DEPTH:
        raise ValueError(f"layout depth must be 0..{LAYOUT_TOKEN_MAX_DEPTH}, got {depth}")
    return f"<NL{depth}>"


def is_layout_token(text: str) -> bool:
    return bool(text and LAYOUT_TOKEN_RE.match(text))


def _line_starts(source: str) -> list[int]:
    starts = [0]
    pos = 0
    for c in source:
        if c == "\n":
            starts.append(pos + 1)
        pos += 1
    return starts


def _offset_in_any_span(offset: int, spans: list[tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= offset < b:
            return True
    return False


def _string_literal_spans(source: str) -> tuple[list[tuple[int, int]], bool]:
    """Character spans of STRING tokens; ok False if tokenize could not run."""
    spans: list[tuple[int, int]] = []
    try:
        readline = io.StringIO(source).readline
        toks = list(tokenize.generate_tokens(readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, MemoryError):
        return [], False

    line_starts = _line_starts(source)

    def to_off(line: int, col: int) -> int:
        if line < 1 or line > len(line_starts):
            return len(source)
        base = line_starts[line - 1]
        return min(len(source), base + col)

    for tok in toks:
        if tok.type != tokenize.STRING:
            continue
        a = to_off(tok.start[0], tok.start[1])
        b = to_off(tok.end[0], tok.end[1])
        if b > a:
            spans.append((a, b))
    return spans, True


def infer_indent_unit(source: str) -> dict[str, Any]:
    """
    Infer indent style and unit from non-empty lines' leading whitespace.

    *mixed* tabs+spaces across the file or within a prefix → usable False.
    Lines inside multi-line string tokens are ignored so docstrings / literals
    do not distort the inferred unit.
    """
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    protected_lines = find_multiline_string_line_spans(text)
    space_widths: list[int] = []
    saw_tab_line = False
    saw_space_line = False

    for i, ln in enumerate(lines):
        line_no = i + 1
        if line_no in protected_lines:
            continue
        if not ln.strip():
            continue
        m = re.match(r"^[ \t]*", ln)
        assert m is not None
        prefix = m.group(0)
        if not prefix:
            continue
        if "\t" in prefix and " " in prefix:
            return {
                "indent_style": "mixed",
                "indent_unit": None,
                "usable": False,
                "reason": "leading prefix mixes tabs and spaces on a line",
            }
        if "\t" in prefix:
            saw_tab_line = True
            if not re.fullmatch(r"\t+", prefix):
                return {
                    "indent_style": "mixed",
                    "indent_unit": None,
                    "usable": False,
                    "reason": "tab prefix contains non-tab characters",
                }
        if " " in prefix:
            saw_space_line = True
            if not re.fullmatch(r" +", prefix):
                return {
                    "indent_style": "mixed",
                    "indent_unit": None,
                    "usable": False,
                    "reason": "space prefix contains non-space characters",
                }
            space_widths.append(len(prefix))

    if saw_tab_line and saw_space_line:
        return {
            "indent_style": "mixed",
            "indent_unit": None,
            "usable": False,
            "reason": "file mixes tab-indented and space-indented lines",
        }

    if saw_tab_line:
        return {
            "indent_style": "tabs",
            "indent_unit": 1,
            "usable": True,
            "reason": "uniform tab indents (one tab per depth unit)",
        }

    if not space_widths:
        return {
            "indent_style": "spaces",
            "indent_unit": 4,
            "usable": True,
            "reason": "no indented non-empty lines; default unit 4 for depth-0 only",
        }

    pos = [w for w in space_widths if w > 0]
    if not pos:
        return {
            "indent_style": "spaces",
            "indent_unit": 4,
            "usable": True,
            "reason": "only zero-width indents; default unit 4",
        }

    g = reduce(gcd, pos)
    if g <= 0:
        g = 4

    def all_div(n: int) -> bool:
        return all(w % n == 0 for w in space_widths)

    unit: int | None
    if all_div(4):
        unit = 4
    elif all_div(2):
        unit = 2
    else:
        unit = g if g >= 1 else 4

    return {
        "indent_style": "spaces",
        "indent_unit": int(unit),
        "usable": True,
        "reason": f"inferred space unit {unit} from non-empty line indents (gcd={g})",
    }


def encode_layout_indentation(source: str) -> dict[str, Any]:
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    info = infer_indent_unit(text)
    used_types: set[str] = set()
    if not info["usable"]:
        return {
            "encoded_text": text,
            "layout_tokens_used": [],
            "indent_unit": info.get("indent_unit"),
            "indent_style": info.get("indent_style", "unknown"),
            "encoded_line_count": 0,
            "skipped_line_count": 0,
            "usable": False,
            "reason": str(info.get("reason", "unusable")),
        }

    spans, tok_ok = _string_literal_spans(text)
    if not tok_ok:
        return {
            "encoded_text": text,
            "layout_tokens_used": [],
            "indent_unit": info["indent_unit"],
            "indent_style": info["indent_style"],
            "encoded_line_count": 0,
            "skipped_line_count": 0,
            "usable": False,
            "reason": "tokenize failed; skip encoding to protect string literals",
        }

    protected_lines = find_multiline_string_line_spans(text)
    lines = text.split("\n")
    if not lines:
        return {
            "encoded_text": text,
            "layout_tokens_used": [],
            "indent_unit": info["indent_unit"],
            "indent_style": info["indent_style"],
            "encoded_line_count": 0,
            "skipped_line_count": 0,
            "usable": True,
            "reason": "empty file",
        }

    line_starts = _line_starts(text)
    out: list[str] = [lines[0]]
    encoded_n = 0
    skipped_n = 0
    indent_unit = int(info["indent_unit"] or 1)
    istyle = str(info["indent_style"])

    for i in range(1, len(lines)):
        line_no = i + 1
        ln = lines[i]
        start_off = line_starts[i] if i < len(line_starts) else len(text)

        if line_no in protected_lines or _offset_in_any_span(start_off, spans):
            out.append("\n")
            out.append(ln)
            skipped_n += 1
            continue

        m = re.match(r"^([ \t]*)(.*)$", ln, re.DOTALL)
        assert m is not None
        leading, rest = m.group(1), m.group(2)

        if "\t" in leading and " " in leading:
            out.append("\n")
            out.append(ln)
            skipped_n += 1
            continue

        depth: int | None
        if istyle == "tabs":
            if leading and not re.fullmatch(r"\t+", leading):
                out.append("\n")
                out.append(ln)
                skipped_n += 1
                continue
            depth = len(leading)
        else:
            if leading and "\t" in leading:
                out.append("\n")
                out.append(ln)
                skipped_n += 1
                continue
            if not leading:
                depth = 0
            else:
                if len(leading) % indent_unit != 0:
                    out.append("\n")
                    out.append(ln)
                    skipped_n += 1
                    continue
                depth = len(leading) // indent_unit

        if depth is None:
            out.append("\n")
            out.append(ln)
            skipped_n += 1
            continue

        if depth > LAYOUT_TOKEN_MAX_DEPTH:
            out.append("\n")
            out.append(ln)
            skipped_n += 1
            continue

        tok = build_layout_token(depth)
        used_types.add(tok)
        out.append(tok)
        out.append(rest)
        encoded_n += 1

    encoded_text = "".join(out)
    layout_tokens_used = sorted(used_types, key=lambda s: int(s[3:-1]))

    return {
        "encoded_text": encoded_text,
        "layout_tokens_used": layout_tokens_used,
        "indent_unit": indent_unit,
        "indent_style": istyle,
        "encoded_line_count": encoded_n,
        "skipped_line_count": skipped_n,
        "usable": True,
        "reason": "encoded newline+indent prefixes where safe",
    }


def decode_layout_indentation(
    encoded_text: str,
    *,
    indent_unit: int,
    indent_style: str,
) -> str:
    istyle = indent_style

    def repl(m: re.Match[str]) -> str:
        d = int(m.group(1))
        if istyle == "tabs":
            return "\n" + "\t" * d
        u = max(1, int(indent_unit))
        return "\n" + " " * (d * u)

    return LAYOUT_TOKEN_FIND_RE.sub(repl, encoded_text)


def verify_layout_roundtrip(source: str) -> dict[str, Any]:
    enc = encode_layout_indentation(source)
    encoded_ok = bool(enc.get("usable"))
    if not encoded_ok:
        return {
            "encoded": False,
            "decoded": False,
            "roundtrip_equal": False,
            "encoded_token_count": 0,
            "decoded_text": "",
            "reason": str(enc.get("reason", "encode skipped")),
        }

    dec = decode_layout_indentation(
        enc["encoded_text"],
        indent_unit=int(enc["indent_unit"] or 1),
        indent_style=str(enc["indent_style"]),
    )
    eq = dec == source.replace("\r\n", "\n").replace("\r", "\n")
    token_count = len(LAYOUT_TOKEN_FIND_RE.findall(enc["encoded_text"]))
    return {
        "encoded": True,
        "decoded": True,
        "roundtrip_equal": eq,
        "encoded_token_count": token_count,
        "decoded_text": dec,
        "reason": "ok" if eq else "round-trip mismatch",
    }


def estimate_layout_encoding_effect(
    baseline_text: str,
    encoded_text: str,
    used_layout_tokens: list[str],
    *,
    tokenizer: Any = None,
    tok_type: str | None = None,
    vocab_cost_mode: str | None = None,
) -> dict[str, Any]:
    mode = vocab_cost_mode or VOCAB_COST_MODE
    base_seq = count_sequence_tokens(
        baseline_text, tokenizer=tokenizer, tok_type=tok_type
    )
    enc_seq = count_sequence_tokens(
        encoded_text, tokenizer=tokenizer, tok_type=tok_type
    )
    seq_net = base_seq - enc_seq
    uniq = sorted(set(used_layout_tokens))
    entries = [{"token": t, "definition": t, "kind": "layout"} for t in uniq]
    vocab_intro = compute_vocab_intro_cost(
        entries, mode=mode, tokenizer=tokenizer, tok_type=tok_type
    )
    effective_net = seq_net - vocab_intro
    return {
        "baseline_sequence_tokens": base_seq,
        "encoded_sequence_tokens": enc_seq,
        "sequence_net_saving": seq_net,
        "vocab_intro_tokens": vocab_intro,
        "effective_total_net_saving": effective_net,
    }


def run_stage2_post_layout_encode(
    text: str,
    *,
    path: str | None = None,
) -> tuple[str, dict[str, Any]]:
    del path  # reserved for future logging / diagnostics
    enc = encode_layout_indentation(text)
    if not enc["usable"]:
        return text, {
            "text": text,
            "layout_encoding_used": False,
            "layout_roundtrip_equal": False,
            "layout_encode_success": False,
            "layout_decode_success": False,
            "layout_tokens_used_count": 0,
            "layout_encoded_line_count": 0,
            "layout_skipped_line_count": int(enc.get("skipped_line_count", 0)),
            "indent_unit": enc.get("indent_unit"),
            "indent_style": enc.get("indent_style"),
            "layout_tokens_used": [],
            "reason": str(enc.get("reason", "")),
        }

    decoded = decode_layout_indentation(
        enc["encoded_text"],
        indent_unit=int(enc["indent_unit"] or 1),
        indent_style=str(enc["indent_style"]),
    )
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    rt_ok = decoded == norm
    used = list(enc.get("layout_tokens_used") or [])
    return enc["encoded_text"], {
        "text": enc["encoded_text"],
        "layout_encoding_used": True,
        "layout_roundtrip_equal": rt_ok,
        "layout_encode_success": True,
        "layout_decode_success": True,
        "layout_tokens_used_count": len(used),
        "layout_encoded_line_count": int(enc["encoded_line_count"]),
        "layout_skipped_line_count": int(enc["skipped_line_count"]),
        "indent_unit": enc["indent_unit"],
        "indent_style": enc["indent_style"],
        "layout_tokens_used": used,
        "reason": "ok" if rt_ok else "round-trip failed",
    }
