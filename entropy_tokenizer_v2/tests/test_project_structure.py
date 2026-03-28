from pathlib import Path


def test_core_files_exist() -> None:
    expected_files = [
        "config.py",
        "marker_count.py",
        "markers.py",
        "pipeline.py",
        "repo_miner.py",
        "syntax_compressor.py",
        "token_scorer.py",
        "lossy_cleaner.py",
        "stage2/__init__.py",
        "stage2/config.py",
        "stage2/cleaning.py",
        "eval/bootstrap_v2.py",
        "eval/run_v2.py",
        "eval/v2_eval.py",
    ]
    for rel in expected_files:
        assert Path(rel).exists(), f"missing required file: {rel}"


def test_key_directories_exist() -> None:
    for rel in ("cache", "results", "data", "docs", "tests"):
        assert Path(rel).exists(), f"missing required directory: {rel}"


def test_gitignore_covers_cache_artifacts() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in gitignore
    assert "*.pyc" in gitignore
