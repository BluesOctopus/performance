from literal_codec.codebook.assigner import GreedyPrefixFreeAssigner
from literal_codec.config import AssignmentConfig, CandidateSearchConfig
from literal_codec.tokenizer.mock_tokenizer import MockTokenizerAdapter
from literal_codec.types import FieldProfile, LiteralStat


def test_assigner_prefers_high_weight_literals():
    profile = FieldProfile(
        field_name="service_name",
        sample_size=10,
        cardinality=2,
        entropy_bits=1.0,
        expected_raw_token_cost=2.4,
        stats=[
            LiteralStat("serve_auth", 8, 0.8, 0.32, 3),
            LiteralStat("debug", 2, 0.2, 2.32, 1),
        ],
    )
    assigner = GreedyPrefixFreeAssigner(
        tokenizer=MockTokenizerAdapter(),
        candidate_config=CandidateSearchConfig(alphabet="abc", max_code_length_chars=2),
        assignment_config=AssignmentConfig(weight_mode="p_times_cost", min_code_token_cost=1),
        escape_prefix="__L__",
    )
    codebook, coded_cost = assigner.build_codebook(profile)
    assert len(codebook.assignments) >= 1
    assert any(a.literal == "serve_auth" for a in codebook.assignments)
    assert coded_cost <= profile.expected_raw_token_cost
