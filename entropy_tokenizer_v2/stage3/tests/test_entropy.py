from literal_codec.stats.entropy import entropy_bits, surprisal_bits


def test_surprisal_bits():
    s = surprisal_bits(0.25)
    assert abs(s - 2.0) < 1e-9


def test_entropy_bits_binary_uniform():
    h = entropy_bits({"a": 0.5, "b": 0.5})
    assert abs(h - 1.0) < 1e-9
