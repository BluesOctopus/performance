"""Bootstrap helpers for eval scripts.

This project now uses stable package import paths, so we no longer need to
modify ``sys.path`` at runtime.
"""
from __future__ import annotations

from pathlib import Path

V2_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = V2_DIR.parent


def ensure() -> None:
    # no-op: kept for backward compatibility
    return


def ensure_with_simpy() -> None:
    # If Simpy is required, it should be installed as a normal dependency.
    # We no longer dynamically mutate ``sys.path``.
    ensure()
    return
