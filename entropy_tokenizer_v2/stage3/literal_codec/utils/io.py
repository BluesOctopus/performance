"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    """Load CSV into a list of dict records."""
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def write_json(path: Path, data: Any) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
