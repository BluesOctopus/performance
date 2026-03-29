"""
Encode / decode Python source with Stage3 Plan A field codebooks.

NAME tokens use a single identifier: ``{escape_prefix}{V|A}{code}`` where *code*
matches ``[A-Za-z0-9_]+``. STRING tokens are replaced with ``repr(f"{escape_prefix}S{code}")``.

Identifiers that already start with *escape_prefix* are escaped as ``{escape_prefix}{name}``
(double prefix) so they remain distinguishable from compressed tokens.

Only STRING spellings present in the mined codebook are replaced (skips f-strings,
multiline strings, and ambiguous cases) to preserve exact roundtrip.
"""

from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from typing import Optional

import builtins

from literal_codec.types import FieldCodebook

from token_scorer import _has_overlap, _line_start_offsets, _pos_to_offset

_PROTECTED = set(keyword.kwlist) | set(dir(builtins)) | {
    "self",
    "cls",
    "__init__",
    "__name__",
    "__main__",
    "True",
    "False",
    "None",
    "args",
    "kwargs",
}

_TAG = {"variable": "V", "attribute": "A", "string": "S"}
_REV_TAG = {v: k for k, v in _TAG.items()}

_NAME_PAYLOAD_RE = re.compile(r"^([VAS])([A-Za-z0-9_]+)$")


def _is_fstring_token(s: str) -> bool:
    t = s.lstrip()
    return t.startswith(
        (
            "f'",
            'f"',
            "f'''",
            'f"""',
            "F'",
            'F"',
            "F'''",
            'F"""',
            "rf'",
            'rf"',
            "fr'",
            'fr"',
            "RF'",
            'RF"',
            "FR'",
            'FR"',
        )
    )


def _lit_to_code_maps(codebooks: dict[str, FieldCodebook]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for field in ("variable", "attribute", "string"):
        cb = codebooks.get(field)
        out[field] = {a.literal: a.code for a in cb.assignments} if cb else {}
    return out


def _code_to_lit_maps(codebooks: dict[str, FieldCodebook]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for field in ("variable", "attribute", "string"):
        cb = codebooks.get(field)
        out[field] = {a.code: a.literal for a in cb.assignments} if cb else {}
    return out


def _encode_name(escape: str, tag: str, code: str) -> str:
    return f"{escape}{tag}{code}"


def _encode_string_token(escape: str, code: str) -> str:
    return repr(f"{escape}S{code}")


def _string_payload_re(escape: str) -> re.Pattern[str]:
    return re.compile(re.escape(escape) + r"S([A-Za-z0-9_]+)$")


def _iter_plan_a_spans(
    text: str,
    *,
    escape: str,
    lit_to_code: dict[str, dict[str, str]],
    protected_spans: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    if not text:
        return []
    line_starts = _line_start_offsets(text)
    spans: list[tuple[int, int, str]] = []

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []

    prev_sig: Optional[tokenize.TokenInfo] = None

    for tok in tokens:
        ttype, tstr = tok.type, tok.string
        repl: Optional[str] = None

        if ttype == tokenize.NAME:
            if tstr not in _PROTECTED:
                if tstr.startswith(escape):
                    repl = escape + tstr
                else:
                    is_attr = (
                        prev_sig is not None
                        and prev_sig.type == tokenize.OP
                        and prev_sig.string == "."
                    )
                    if is_attr:
                        code = lit_to_code.get("attribute", {}).get(tstr)
                        if code:
                            repl = _encode_name(escape, _TAG["attribute"], code)
                    else:
                        code = lit_to_code.get("variable", {}).get(tstr)
                        if code:
                            repl = _encode_name(escape, _TAG["variable"], code)

        elif ttype == tokenize.STRING:
            if not _is_fstring_token(tstr) and "\n" not in tstr and "\r" not in tstr:
                code = lit_to_code.get("string", {}).get(tstr)
                if code:
                    repl = _encode_string_token(escape, code)

        if repl is not None:
            start = _pos_to_offset(line_starts, tok.start)
            end = _pos_to_offset(line_starts, tok.end)
            if not _has_overlap(start, end, protected_spans):
                spans.append((start, end, repl))

        if ttype != tokenize.NL and ttype != tokenize.COMMENT and ttype != tokenize.ENDMARKER:
            prev_sig = tok

    return spans


def _apply_replacements(text: str, spans: list[tuple[int, int, str]]) -> str:
    out = text
    for start, end, repl in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:start] + repl + out[end:]
    return out


def encode_python_source_plan_a(
    text: str,
    codebooks: dict[str, FieldCodebook],
    *,
    escape_prefix: str,
) -> str:
    from markers import get_syn_line_spans

    if not codebooks:
        return text
    lit_to_code = _lit_to_code_maps(codebooks)
    protected = get_syn_line_spans(text)
    spans = _iter_plan_a_spans(
        text, escape=escape_prefix, lit_to_code=lit_to_code, protected_spans=protected
    )
    return _apply_replacements(text, spans)


def _decode_name_token(name: str, escape: str, code_to_lit: dict[str, dict[str, str]]) -> Optional[str]:
    if not name.startswith(escape):
        return None
    payload = name[len(escape) :]
    if payload.startswith(escape):
        return payload
    m = _NAME_PAYLOAD_RE.match(payload)
    if not m:
        return None
    tag, code = m.group(1), m.group(2)
    field = _REV_TAG.get(tag)
    if field is None:
        return None
    return code_to_lit.get(field, {}).get(code)


def _decode_string_token(
    tok_text: str,
    escape: str,
    code_to_lit: dict[str, dict[str, str]],
) -> Optional[str]:
    try:
        inner = ast.literal_eval(tok_text)
    except (SyntaxError, ValueError, MemoryError):
        return None
    if not isinstance(inner, str):
        return None
    m = _string_payload_re(escape).match(inner)
    if not m:
        return None
    code = m.group(1)
    return code_to_lit.get("string", {}).get(code)


def decode_python_source_plan_a(
    text: str,
    codebooks: dict[str, FieldCodebook],
    *,
    escape_prefix: str,
) -> str:
    """Inverse of :func:`encode_python_source_plan_a`."""
    if not codebooks:
        return text
    code_to_lit = _code_to_lit_maps(codebooks)
    line_starts = _line_start_offsets(text)
    spans: list[tuple[int, int, str]] = []

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text

    for tok in tokens:
        ttype, tstr = tok.type, tok.string
        repl: Optional[str] = None
        if ttype == tokenize.NAME:
            dec = _decode_name_token(tstr, escape_prefix, code_to_lit)
            if dec is not None:
                repl = dec
        elif ttype == tokenize.STRING:
            dec = _decode_string_token(tstr, escape_prefix, code_to_lit)
            if dec is not None:
                repl = dec
        if repl is not None:
            start = _pos_to_offset(line_starts, tok.start)
            end = _pos_to_offset(line_starts, tok.end)
            spans.append((start, end, repl))

    return _apply_replacements(text, spans)


def verify_roundtrip_plan_a(
    text: str,
    codebooks: dict[str, FieldCodebook],
    *,
    escape_prefix: str,
) -> bool:
    mid = encode_python_source_plan_a(text, codebooks, escape_prefix=escape_prefix)
    back = decode_python_source_plan_a(mid, codebooks, escape_prefix=escape_prefix)
    return back == text
