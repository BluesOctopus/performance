"""Post-prune Plan A assignments where corpus sequence savings do not cover vocab intro."""

from __future__ import annotations

from typing import Any

from ..types import CodeAssignment, FieldBuildResult, FieldCodebook


def _single_entry_vocab_intro_tokens(
    field: str,
    literal: str,
    code: str,
    escape_prefix: str,
    tokenizer: Any,
    tok_type: str,
) -> int:
    from config import VOCAB_COST_MODE
    from placeholder_accounting import compute_vocab_intro_cost
    from .surface_cost import surface_text_for_code

    surface = surface_text_for_code(field, code, escape_prefix)
    entries = [
        {
            "token": surface,
            "kind": "stage3_plan_a_prune",
            "definition": literal,
            "field": field,
            "code": code,
        }
    ]
    return int(
        compute_vocab_intro_cost(
            entries,
            mode=VOCAB_COST_MODE,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
    )


def prune_plan_a_field_results(
    field_results: list[FieldBuildResult],
    codebooks: dict[str, FieldCodebook],
    *,
    tokenizer: Any,
    tok_type: str,
    escape_prefix: str,
    enabled: bool,
) -> tuple[dict[str, FieldCodebook], list[FieldBuildResult], dict[str, Any]]:
    """
    Remove assignments with non-positive net gain:
    ``count * max(0, raw - encoded) <= vocab_intro_tokens(entry)``.
    """
    if not enabled:
        return codebooks, field_results, {"stage3_plan_a_post_prune_enabled": False}

    removed: list[dict[str, Any]] = []
    pre = sum(len(cb.assignments) for cb in codebooks.values())
    new_books: dict[str, FieldCodebook] = {}

    for fr in field_results:
        fname = fr.profile.field_name
        book = codebooks[fname]
        stat_by_lit = {s.literal: s for s in fr.profile.stats}
        kept: list[CodeAssignment] = []
        for a in book.assignments:
            st = stat_by_lit.get(a.literal)
            if st is None:
                kept.append(a)
                continue
            delta = max(0, st.raw_token_cost - a.code_token_cost)
            seq_save = st.count * delta
            intro = _single_entry_vocab_intro_tokens(
                fname, a.literal, a.code, escape_prefix, tokenizer, tok_type
            )
            if seq_save > intro:
                kept.append(a)
            else:
                removed.append(
                    {
                        "field": fname,
                        "literal": a.literal,
                        "code": a.code,
                        "count": st.count,
                        "seq_save": seq_save,
                        "intro_tokens": intro,
                    }
                )
        meta = dict(book.metadata or {})
        meta["post_pruned"] = True
        meta["pre_prune_assignments"] = len(book.assignments)
        meta["post_prune_assignments"] = len(kept)
        new_books[fname] = FieldCodebook(
            field_name=book.field_name,
            version=book.version,
            assignments=kept,
            escape_prefix=book.escape_prefix,
            metadata=meta,
        )

    new_frs: list[FieldBuildResult] = []
    for fr in field_results:
        fname = fr.profile.field_name
        nb = new_books[fname]
        cov = len(nb.assignments) / fr.profile.cardinality if fr.profile.cardinality else 0.0
        new_frs.append(
            FieldBuildResult(
                profile=fr.profile,
                codebook=nb,
                expected_coded_token_cost=fr.expected_coded_token_cost,
                theoretical_headroom=fr.theoretical_headroom,
                dictionary_coverage=cov,
                total_expected_gain=fr.total_expected_gain,
            )
        )

    post = sum(len(cb.assignments) for cb in new_books.values())
    report = {
        "stage3_plan_a_post_prune_enabled": True,
        "stage3_plan_a_post_prune_pre_assignments": pre,
        "stage3_plan_a_post_prune_post_assignments": post,
        "stage3_plan_a_post_prune_removed_count": len(removed),
        "stage3_plan_a_post_prune_removed_json": removed[:200],
    }
    return new_books, new_frs, report
