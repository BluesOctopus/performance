from pathlib import Path

from marker_count import count_augmented
from markers import RE_ALL_MARKERS
from tests.fixtures import SimpleOfflineTokenizer


def test_fake_tokenizer_augmented_count_is_stable() -> None:
    text = "<SYN_0> foo\nfoo = 123\n"
    tok = SimpleOfflineTokenizer()
    c1 = count_augmented(text, tok, "hf", pattern=RE_ALL_MARKERS)
    c2 = count_augmented(text, tok, "hf", pattern=RE_ALL_MARKERS)
    assert c1 == c2


def test_verify_script_does_not_depend_on_cache_or_results_content() -> None:
    verify_src = Path("scripts/verify.py").read_text(encoding="utf-8")
    assert "cache/" not in verify_src
    assert "results/" not in verify_src
