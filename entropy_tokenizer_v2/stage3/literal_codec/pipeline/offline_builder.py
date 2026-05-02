"""Offline codebook builder CLI and APIs."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..codebook.assigner import GreedyPrefixFreeAssigner
from ..codebook.decoder import FieldDecoder
from ..codebook.encoder import FieldEncoder
from ..codebook.models import codebook_to_dict
from ..config import CompressionConfig
from .report import field_report, summary_report
from ..stats.field_profile import FieldProfiler
from ..tokenizer.base import TokenizerAdapter
from ..tokenizer.mock_tokenizer import MockTokenizerAdapter
from ..tokenizer.optional_tiktoken_adapter import OptionalTiktokenAdapter
from ..types import FieldBuildResult, FieldCodebook
from ..utils.io import read_csv_records, write_json
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OfflineBuildArtifacts:
    """Build outputs and convenience encode/decode methods."""

    field_results: list[FieldBuildResult]
    codebooks: dict[str, FieldCodebook]
    report: dict[str, Any]

    def encode_record(self, record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record)
        for field, book in self.codebooks.items():
            if field not in out:
                continue
            out[field] = FieldEncoder(book).encode_value(str(out[field]))
        return out

    def decode_record(self, record: dict[str, Any], strict: bool = True) -> dict[str, Any]:
        out = dict(record)
        for field, book in self.codebooks.items():
            if field not in out:
                continue
            out[field] = FieldDecoder(book, strict=strict).decode_value(str(out[field]))
        return out

    def save(self, output_dir: Path) -> None:
        codebook_payload = {
            "fields": {name: codebook_to_dict(cb) for name, cb in self.codebooks.items()}
        }
        write_json(output_dir / "codebook.json", codebook_payload)
        write_json(output_dir / "report.json", self.report)


class OfflineCodebookBuilder:
    """Main Stage3 Plan-A builder."""

    def __init__(self, config: CompressionConfig, tokenizer: TokenizerAdapter | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or MockTokenizerAdapter()
        self.profiler = FieldProfiler(tokenizer=self.tokenizer, smoothing=self.config.smoothing)
        self.assigner = GreedyPrefixFreeAssigner(
            tokenizer=self.tokenizer,
            candidate_config=self.config.candidate_search,
            assignment_config=self.config.assignment,
            escape_prefix=self.config.escape_prefix,
            version=self.config.codebook_version,
        )

    def _build_one_field(self, field_name: str, values: list[str]) -> FieldBuildResult:
        profile = self.profiler.build(field_name=field_name, values=values)
        codebook, expected_coded_cost = self.assigner.build_codebook(profile)
        total_gain = profile.expected_raw_token_cost - expected_coded_cost
        coverage = (
            len(codebook.assignments) / profile.cardinality if profile.cardinality > 0 else 0.0
        )
        theoretical_headroom = max(0.0, profile.expected_raw_token_cost - profile.entropy_bits)
        return FieldBuildResult(
            profile=profile,
            codebook=codebook,
            expected_coded_token_cost=expected_coded_cost,
            theoretical_headroom=theoretical_headroom,
            dictionary_coverage=coverage,
            total_expected_gain=total_gain,
        )

    def build_from_records(self, records: list[dict[str, Any]], fields: list[str]) -> OfflineBuildArtifacts:
        field_results: list[FieldBuildResult] = []
        codebooks: dict[str, FieldCodebook] = {}

        for field in fields:
            values = [str(item[field]) for item in records if field in item and item[field] is not None]
            logger.info("building field=%s sample_size=%s", field, len(values))
            result = self._build_one_field(field, values)
            field_results.append(result)
            codebooks[field] = result.codebook

        report_fields = [field_report(r) for r in field_results]
        report = {
            "fields": report_fields,
            "summary": summary_report(field_results),
            "assumptions": {
                "plan": "A",
                "semantic_loss": 0.0,
                "tokenizer": self.tokenizer.__class__.__name__,
            },
        }
        return OfflineBuildArtifacts(field_results=field_results, codebooks=codebooks, report=report)

    def build_from_csv(self, csv_path: Path, fields: list[str]) -> OfflineBuildArtifacts:
        records = read_csv_records(csv_path)
        return self.build_from_records(records=records, fields=fields)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage3 literal compression codebook.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    parser.add_argument("--fields", nargs="+", required=True, help="Field names to compress.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output artifact directory.")
    parser.add_argument("--escape_prefix", default="__L__", type=str)
    parser.add_argument("--alpha", default=1.0, type=float, help="Lidstone alpha.")
    parser.add_argument("--max_code_length_chars", default=4, type=int)
    parser.add_argument("--oversubscribe_factor", default=8, type=int)
    parser.add_argument("--weight_mode", default="p_times_cost", choices=["p_times_cost", "p_only"])
    parser.add_argument("--use_tiktoken", action="store_true", help="Use tiktoken if installed.")
    parser.add_argument("--log_level", default="INFO", type=str)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = CompressionConfig(fields=args.fields, escape_prefix=args.escape_prefix)
    config.smoothing.alpha = args.alpha
    config.candidate_search.max_code_length_chars = args.max_code_length_chars
    config.candidate_search.oversubscribe_factor = args.oversubscribe_factor
    config.assignment.weight_mode = args.weight_mode

    tokenizer: TokenizerAdapter
    if args.use_tiktoken:
        tokenizer = OptionalTiktokenAdapter()
        logger.info("tokenizer=%s fallback=%s", tokenizer.__class__.__name__, tokenizer.is_fallback)
    else:
        tokenizer = MockTokenizerAdapter()

    builder = OfflineCodebookBuilder(config=config, tokenizer=tokenizer)
    artifacts = builder.build_from_csv(args.input, fields=list(args.fields))
    artifacts.save(args.output_dir)
    logger.info("saved artifacts to %s", args.output_dir)


if __name__ == "__main__":
    main()
