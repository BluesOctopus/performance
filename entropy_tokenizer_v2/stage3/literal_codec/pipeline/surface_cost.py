"""
Real **surface form** token costs for Plan A (must match ``source_codec`` output).

* variable / attribute NAME replacements: ``{escape}{V|A}{code}``
* string replacements: ``repr(f"{escape}S{code}")`` — same as ``_encode_string_token``.
"""

from __future__ import annotations

from literal_codec.tokenizer.base import TokenizerAdapter

FIELD_TAG = {"variable": "V", "attribute": "A", "string": "S"}


def surface_text_for_code(field_name: str, code: str, escape_prefix: str) -> str:
    """
    Exact text that appears in the compressed Python source for this assignment.

    For *string* field this is a full STRING token spelling (includes quotes), matching
    ``source_codec._encode_string_token``.

    Offline / generic field names (e.g. ``service_name`` from CSV demos) are treated like
    NAME replacements: ``{escape_prefix}V{code}``, same as ``variable``.
    """
    if field_name == "string":
        return repr(f"{escape_prefix}S{code}")
    tag = FIELD_TAG.get(field_name)
    if tag is None:
        tag = "V"
    return f"{escape_prefix}{tag}{code}"


def encoded_form_token_cost(
    field_name: str,
    code: str,
    escape_prefix: str,
    tokenizer: TokenizerAdapter,
) -> int:
    """Tokenizer length of the real compressed surface form."""
    surface = surface_text_for_code(field_name, code, escape_prefix)
    return tokenizer.token_length(surface)
