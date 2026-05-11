"""Unified offline verification entrypoint."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

KEY_REGRESSION_FILES = [
    "tests/test_stage3_safe_replacement.py",
    "tests/test_stage3_skip_syn.py",
    "tests/test_stage1_greedy_selection.py",
    "tests/test_pipeline_smoke.py",
]


def _parse_counts(output: str) -> tuple[int, int]:
    passed = 0
    failed = 0

    m_passed = re.search(r"(\d+)\s+passed", output)
    if m_passed:
        passed = int(m_passed.group(1))

    m_failed = re.search(r"(\d+)\s+failed", output)
    if m_failed:
        failed = int(m_failed.group(1))

    return passed, failed


def _parse_failed_tests(output: str) -> list[str]:
    return re.findall(r"^FAILED\s+([^\s]+)\s+-", output, flags=re.MULTILINE)


def main() -> int:
    start = time.perf_counter()
    cmd = [sys.executable, "-m", "pytest", "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - start

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    passed, failed = _parse_counts(combined)
    total = passed + failed
    failed_tests = _parse_failed_tests(combined)
    success = proc.returncode == 0 and failed == 0

    report = {
        "success": success,
        "python_version": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "failed_tests": failed_tests,
        "duration_sec": round(duration, 3),
    }
    report_path = Path("verification_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_written = report_path.exists()

    print(f"[verify] python={report['python_version']}")
    print(f"[verify] cwd={report['cwd']}")
    print(f"[verify] command: {' '.join(cmd)}")
    print(
        f"[verify] success={report['success']} total={total} "
        f"passed={passed} failed={failed} duration_sec={report['duration_sec']}"
    )
    print(f"[verify] key_regressions={', '.join(KEY_REGRESSION_FILES)}")
    print(f"[verify] report_written={report_written} report={report_path.resolve()}")
    if failed_tests:
        print(f"[verify] failed_tests_summary={failed_tests[:10]}")
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
