"""Insert ``entropy_tokenizer_v2`` (and optionally Simpy) on ``sys.path`` for eval scripts."""
from __future__ import annotations

import sys
from pathlib import Path

V2_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = V2_DIR.parent


def ensure() -> None:
    p = str(V2_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_with_simpy() -> None:
    ensure()
    from config import SIMPY_DIR

    s = str(SIMPY_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)
