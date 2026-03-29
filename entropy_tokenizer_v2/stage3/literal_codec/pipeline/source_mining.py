"""Build Plan A codebooks from Python source corpora (category = field)."""

from __future__ import annotations

import ast
import io
import keyword
import logging
import tokenize
from dataclasses import dataclass
from typing import Any

import builtins

from literal_codec.codebook.models import codebook_to_dict
from literal_codec.codebook.assigner import GreedyPrefixFreeAssigner
from literal_codec.config import AssignmentConfig, CandidateSearchConfig, CompressionConfig, SmoothingConfig
from literal_codec.stats.field_profile import FieldProfiler
from literal_codec.tokenizer.base import TokenizerAdapter
from literal_codec.types import FieldBuildResult, FieldCodebook

from literal_codec.pipeline.report import field_report, summary_report
from literal_codec.pipeline.v2_token_adapter import V2TokenizerAdapter

logger = logging.getLogger(__name__)

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

PLAN_A_FIELD_ORDER = ("variable", "attribute", "string")


def _safe_tokenize(source: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


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


def collect_category_literal_streams(
    sources: list[str],
    *,
    enabled_categories: frozenset[str],
) -> dict[str, list[str]]:
    """
    For each enabled category, collect token *exact* spellings (with repetition)
    for empirical distribution, mirroring token_scorer vocabulary semantics.
    """
    out: dict[str, list[str]] = {c: [] for c in PLAN_A_FIELD_ORDER if c in enabled_categories}
    for src in sources:
        prev_is_dot = False
        for tok in _safe_tokenize(src):
            ttype, tstr = tok.type, tok.string

            if ttype == tokenize.NAME:
                if tstr not in _PROTECTED:
                    if "variable" in enabled_categories:
                        out.setdefault("variable", []).append(tstr)
                    if prev_is_dot and "attribute" in enabled_categories:
                        out.setdefault("attribute", []).append(tstr)
                prev_is_dot = False

            elif ttype == tokenize.STRING and "string" in enabled_categories:
                if _is_fstring_token(tstr):
                    prev_is_dot = False
                    continue
                if "\n" in tstr or "\r" in tstr:
                    prev_is_dot = False
                    continue
                try:
                    inner = ast.literal_eval(tstr)
                except (SyntaxError, ValueError, MemoryError):
                    prev_is_dot = False
                    continue
                if not isinstance(inner, str):
                    prev_is_dot = False
                    continue
                if inner.startswith("__L__"):
                    prev_is_dot = False
                    continue
                out.setdefault("string", []).append(tstr)
                prev_is_dot = False

            elif ttype == tokenize.OP:
                prev_is_dot = tstr == "."

            else:
                prev_is_dot = False

    return out


def build_compression_config_for_plan_a(
    *,
    escape_prefix: str,
    codebook_version: str,
    min_gain: float,
    enabled_categories: tuple[str, ...],
    use_tiktoken: bool,
) -> CompressionConfig:
    """Build literal_codec CompressionConfig from v2 env-style knobs."""
    cats = frozenset(x.strip() for x in enabled_categories if x.strip())
    # Keep candidate codes identifier-safe; exclude chars used in compressed NAME layout.
    alphabet = "".join(
        ch
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        if ch not in {"|"}
    )
    cfg = CompressionConfig(
        random_seed=7,
        codebook_version=codebook_version,
        escape_prefix=escape_prefix,
        smoothing=SmoothingConfig(method="lidstone", alpha=1.0),
        candidate_search=CandidateSearchConfig(
            alphabet=alphabet,
            max_code_length_chars=4,
            oversubscribe_factor=8,
            max_nodes_to_expand=20000,
        ),
        assignment=AssignmentConfig(
            weight_mode="p_times_cost",
            min_code_token_cost=1,
            min_gain=min_gain,
        ),
        fields=tuple(c for c in PLAN_A_FIELD_ORDER if c in cats),
    )
    del use_tiktoken  # resolved by caller when constructing adapter
    return cfg


def select_tokenizer_adapter(
    tokenizer: Any,
    tok_type: str,
    *,
    prefer_tiktoken: bool,
) -> TokenizerAdapter:
    """
    Always use the same concrete tokenizer as v2 pipeline counting (V2TokenizerAdapter).

    *prefer_tiktoken* is accepted for API compatibility; codebook token costs must match
    the tokenizer used at eval time, so we do not swap to a different tiktoken model here.
    """
    del prefer_tiktoken
    return V2TokenizerAdapter(tokenizer=tokenizer, tok_type=tok_type)


@dataclass(slots=True)
class PlanAMiningResult:
    """Mined Plan A artifacts for RepoConfig."""

    codebooks: dict[str, FieldCodebook]
    field_results: list[FieldBuildResult]
    report: dict[str, Any]
    enabled_categories: tuple[str, ...]


def mine_plan_a_from_sources(
    sources: list[str],
    tokenizer: Any,
    tok_type: str,
    *,
    escape_prefix: str,
    codebook_version: str,
    min_gain: float,
    enabled_categories: tuple[str, ...],
    use_tiktoken: bool,
) -> PlanAMiningResult:
    cats = frozenset(x.strip() for x in enabled_categories if x.strip())
    cfg = build_compression_config_for_plan_a(
        escape_prefix=escape_prefix,
        codebook_version=codebook_version,
        min_gain=min_gain,
        enabled_categories=tuple(cats),
        use_tiktoken=use_tiktoken,
    )
    adapter = select_tokenizer_adapter(tokenizer, tok_type, prefer_tiktoken=use_tiktoken)
    streams = collect_category_literal_streams(sources, enabled_categories=cats)

    profiler = FieldProfiler(tokenizer=adapter, smoothing=cfg.smoothing)
    assigner = GreedyPrefixFreeAssigner(
        tokenizer=adapter,
        candidate_config=cfg.candidate_search,
        assignment_config=cfg.assignment,
        escape_prefix=cfg.escape_prefix,
        version=cfg.codebook_version,
    )

    field_results: list[FieldBuildResult] = []
    codebooks: dict[str, FieldCodebook] = {}

    for field in cfg.fields:
        values = streams.get(field, [])
        logger.info("plan_a mine field=%s n=%s", field, len(values))
        profile = profiler.build(field_name=field, values=values)
        book, _coded = assigner.build_codebook(profile)
        total_gain = profile.expected_raw_token_cost - _coded
        coverage = len(book.assignments) / profile.cardinality if profile.cardinality else 0.0
        theoretical_headroom = max(0.0, profile.expected_raw_token_cost - profile.entropy_bits)
        fr = FieldBuildResult(
            profile=profile,
            codebook=book,
            expected_coded_token_cost=_coded,
            theoretical_headroom=theoretical_headroom,
            dictionary_coverage=coverage,
            total_expected_gain=total_gain,
        )
        field_results.append(fr)
        codebooks[field] = book

    report_fields = [field_report(r) for r in field_results]
    report = {
        "fields": report_fields,
        "summary": summary_report(field_results),
        "assumptions": {
            "plan": "A",
            "semantic_loss": 0.0,
            "tokenizer_adapter": adapter.__class__.__name__,
        },
    }
    return PlanAMiningResult(
        codebooks=codebooks,
        field_results=field_results,
        report=report,
        enabled_categories=tuple(cfg.fields),
    )


def serialize_plan_a_codebooks(codebooks: dict[str, FieldCodebook]) -> dict[str, Any]:
    return {name: codebook_to_dict(cb) for name, cb in codebooks.items()}


def deserialize_plan_a_codebooks(data: dict[str, Any]) -> dict[str, FieldCodebook]:
    from literal_codec.codebook.models import codebook_from_dict

    return {k: codebook_from_dict(v) for k, v in data.items()}


def plan_a_summary_dict(result: PlanAMiningResult) -> dict[str, Any]:
    assignments = sum(len(cb.assignments) for cb in result.codebooks.values())
    covs = [fr.dictionary_coverage for fr in result.field_results if fr.profile.cardinality]
    avg_cov = sum(covs) / len(covs) if covs else 0.0
    gains = [fr.total_expected_gain for fr in result.field_results]
    return {
        "stage3_plan_a_assignments_count": assignments,
        "stage3_plan_a_fields": list(result.enabled_categories),
        "stage3_plan_a_dictionary_coverage_mean": avg_cov,
        "stage3_plan_a_total_expected_gain_sum": sum(gains),
    }
