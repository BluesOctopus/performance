from pathlib import Path


def test_stage2_cleaning_uses_shared_syn_detector() -> None:
    text = Path("stage2/cleaning.py").read_text(encoding="utf-8")
    assert "from markers import is_syn_line" in text
    assert "_SYN_LINE_RE" not in text
    assert "re.compile(" not in text


def test_v2_eval_uses_pipeline_not_local_syn_regex() -> None:
    text = Path("eval/v2_eval.py").read_text(encoding="utf-8")
    assert "from pipeline import" in text
    assert "_SYN_LINE_RE" not in text
