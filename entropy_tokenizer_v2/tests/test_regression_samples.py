from types import SimpleNamespace

from marker_count import count_augmented
from markers import RE_ALL_MARKERS
from pipeline import apply_pipeline
from tests.fixtures import SimpleOfflineTokenizer
from tests.regression_fixtures import REGRESSION_CASES


def _mini_repo_config(rmap: dict[str, str]):
    return SimpleNamespace(
        replacement_map=rmap,
        skeleton_candidates=lambda: [],
    )


def test_regression_cases_are_deterministic() -> None:
    tokenizer = SimpleOfflineTokenizer()
    for case in REGRESSION_CASES:
        repo_cfg = _mini_repo_config(case["rmap"])
        out1, breakdown1 = apply_pipeline(
            case["source"], repo_cfg, tokenizer=tokenizer, tok_type="hf"
        )
        out2, breakdown2 = apply_pipeline(
            case["source"], repo_cfg, tokenizer=tokenizer, tok_type="hf"
        )
        assert out1 == out2, case["name"]
        assert breakdown1 == breakdown2, case["name"]


def test_regression_cases_have_stable_augmented_count() -> None:
    tokenizer = SimpleOfflineTokenizer()
    for case in REGRESSION_CASES:
        c1 = count_augmented(case["source"], tokenizer, "hf", pattern=RE_ALL_MARKERS)
        c2 = count_augmented(case["source"], tokenizer, "hf", pattern=RE_ALL_MARKERS)
        assert c1 == c2, case["name"]
