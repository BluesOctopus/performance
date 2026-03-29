from pathlib import Path

from literal_codec.codebook.assigner import GreedyPrefixFreeAssigner
from literal_codec.config import AssignmentConfig, CandidateSearchConfig, CompressionConfig
from literal_codec.pipeline.offline_builder import OfflineCodebookBuilder
from literal_codec.tokenizer.base import TokenizerAdapter
from literal_codec.tokenizer.mock_tokenizer import MockTokenizerAdapter
from literal_codec.tokenizer.optional_tiktoken_adapter import OptionalTiktokenAdapter
from literal_codec.types import FieldProfile, LiteralStat


class OneTokenTokenizer(TokenizerAdapter):
    def token_length(self, text: str) -> int:
        return 1

    def tokenize(self, text: str) -> list[str]:
        return [text]


def test_build_codebook_and_positive_gain():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "examples" / "sample_literals.csv"
    config = CompressionConfig(fields=["service_name", "env", "tag_prefix"])
    builder = OfflineCodebookBuilder(config=config, tokenizer=MockTokenizerAdapter())
    artifacts = builder.build_from_csv(csv_path, fields=list(config.fields))
    summary = artifacts.report["summary"]
    assert summary["total_expected_coded_tokens"] < summary["total_expected_raw_tokens"]

    row = {"service_name": "serve_auth", "env": "prod", "tag_prefix": "api_v1"}
    encoded = artifacts.encode_record(row)
    decoded = artifacts.decode_record(encoded)
    assert decoded == row


def test_empty_and_single_value_field():
    builder = OfflineCodebookBuilder(config=CompressionConfig(fields=[]), tokenizer=MockTokenizerAdapter())
    artifacts = builder.build_from_records(
        records=[{"f_single": "x"}, {"f_single": "x"}, {"f_single": "x"}],
        fields=["f_empty", "f_single"],
    )
    by_name = {f.profile.field_name: f for f in artifacts.field_results}
    assert by_name["f_empty"].profile.sample_size == 0
    assert by_name["f_single"].profile.cardinality == 1


def test_all_low_gain_literals_not_assigned():
    profile = FieldProfile(
        field_name="f",
        sample_size=2,
        cardinality=2,
        entropy_bits=1.0,
        expected_raw_token_cost=1.0,
        stats=[
            LiteralStat("a", 1, 0.5, 1.0, 1),
            LiteralStat("b", 1, 0.5, 1.0, 1),
        ],
    )
    assigner = GreedyPrefixFreeAssigner(
        tokenizer=OneTokenTokenizer(),
        candidate_config=CandidateSearchConfig(alphabet="ab", max_code_length_chars=1),
        assignment_config=AssignmentConfig(min_code_token_cost=1),
        escape_prefix="__L__",
    )
    codebook, coded_cost = assigner.build_codebook(profile)
    assert len(codebook.assignments) == 0
    assert coded_cost == profile.expected_raw_token_cost


def test_candidate_pool_insufficient_handled():
    profile = FieldProfile(
        field_name="f",
        sample_size=4,
        cardinality=4,
        entropy_bits=2.0,
        expected_raw_token_cost=3.0,
        stats=[
            LiteralStat("serve_auth", 1, 0.25, 2.0, 3),
            LiteralStat("serve_user", 1, 0.25, 2.0, 3),
            LiteralStat("serve_pay", 1, 0.25, 2.0, 3),
            LiteralStat("serve_order", 1, 0.25, 2.0, 3),
        ],
    )
    assigner = GreedyPrefixFreeAssigner(
        tokenizer=MockTokenizerAdapter(),
        candidate_config=CandidateSearchConfig(alphabet="a", max_code_length_chars=1),
        assignment_config=AssignmentConfig(min_code_token_cost=1),
        escape_prefix="__L__",
    )
    codebook, _ = assigner.build_codebook(profile)
    assert len(codebook.assignments) <= 1


def test_optional_tokenizer_fallback():
    adapter = OptionalTiktokenAdapter(model_name="non-existent-model")
    assert adapter.token_length("serve_auth") >= 1
