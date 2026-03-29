from literal_codec.codebook.candidate_pool import CandidatePoolGenerator
from literal_codec.config import CandidateSearchConfig
from literal_codec.tokenizer.mock_tokenizer import MockTokenizerAdapter


def test_candidate_pool_order_and_reservation():
    cfg = CandidateSearchConfig(
        alphabet="ab",
        max_code_length_chars=3,
        oversubscribe_factor=3,
        max_nodes_to_expand=100,
    )
    gen = CandidatePoolGenerator(tokenizer=MockTokenizerAdapter(), config=cfg)
    pool = gen.generate(needed=4, escape_prefix="__", reserved_strings={"a"})
    assert pool
    assert all(item.code != "a" for item in pool)
    sorted_pool = sorted(pool, key=lambda x: (x.token_cost, len(x.code), x.code))
    assert pool == sorted_pool
