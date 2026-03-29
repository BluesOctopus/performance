from literal_codec.stats.frequency import empirical_distribution, lidstone_distribution


def test_empirical_distribution():
    counts = {"a": 3, "b": 1}
    dist = empirical_distribution(counts)  # type: ignore[arg-type]
    assert dist["a"] == 0.75
    assert dist["b"] == 0.25


def test_lidstone_distribution():
    counts = {"x": 2, "y": 2}
    dist = lidstone_distribution(counts, alpha=1.0)  # type: ignore[arg-type]
    assert abs(dist["x"] - 0.5) < 1e-9
    assert abs(dist["y"] - 0.5) < 1e-9
