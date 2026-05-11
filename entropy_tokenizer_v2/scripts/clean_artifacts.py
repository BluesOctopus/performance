"""Clean local runtime artifacts for a delivery-ready workspace."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _remove_pycache_and_pyc() -> list[str]:
    removed: list[str] = []
    for pycache in ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)
            removed.append(str(pycache.relative_to(ROOT)))
    for pyc in ROOT.rglob("*.pyc"):
        if pyc.is_file():
            pyc.unlink(missing_ok=True)
            removed.append(str(pyc.relative_to(ROOT)))
    return removed


def _remove_paths(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p.relative_to(ROOT)))
        elif p.exists():
            p.unlink(missing_ok=True)
            removed.append(str(p.relative_to(ROOT)))
    return removed


def _clean_generated_dir(dir_name: str) -> list[str]:
    removed: list[str] = []
    target = ROOT / dir_name
    if not target.exists():
        return removed
    for child in target.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        removed.append(str(child.relative_to(ROOT)))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean runtime artifacts.")
    parser.add_argument(
        "--with-generated",
        action="store_true",
        help="Also remove generated content under cache/ and results/.",
    )
    args = parser.parse_args()

    removed_items: list[str] = []
    removed_items.extend(_remove_pycache_and_pyc())
    removed_items.extend(
        _remove_paths(
            [
                ROOT / ".pytest_cache",
                ROOT / ".benchmarks",
                ROOT / "verification_report.json",
            ]
        )
    )
    if args.with_generated:
        removed_items.extend(_clean_generated_dir("cache"))
        removed_items.extend(_clean_generated_dir("results"))

    print(f"[clean] removed={len(removed_items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
