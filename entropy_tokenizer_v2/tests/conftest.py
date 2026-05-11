"""Test configuration.

We rely on stable import paths (e.g. `eval.v2_eval`, `stage3.*`) and do not
modify `sys.path` at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _project_test_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
