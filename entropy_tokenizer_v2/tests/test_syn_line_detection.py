from markers import is_syn_line


def test_syn_line_detection_accepts_valid_lines() -> None:
    assert is_syn_line("<SYN_0> foo bar")
    assert is_syn_line("    <SYN_12>")


def test_syn_line_detection_rejects_invalid_lines() -> None:
    assert not is_syn_line("<SYN_X>")
    assert not is_syn_line("foo <SYN_0>")
