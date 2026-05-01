from __future__ import annotations

from pathlib import Path

from offline_chunks import build_chunks, extract_python_chunks


class FakeEncoder:
    def encode(self, text: str):
        return text.split()


def test_extract_python_chunks_ast_mode(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "class A:\n"
        "    def method(self, x):\n"
        "        return x\n"
        "\n"
        "def top(y):\n"
        "    return y + 1\n",
        encoding="utf-8",
    )

    records = extract_python_chunks(sample, root=tmp_path, encoder=FakeEncoder())
    kinds = {(item["symbol_type"], item["symbol_name"]) for item in records}

    assert ("module", "sample") in kinds
    assert ("class", "A") in kinds
    assert ("function", "A.method") in kinds
    assert ("function", "top") in kinds
    assert all(item["ast_parse_ok"] is True for item in records)
    assert all(item["raw_tokens"] > 0 for item in records)


def test_extract_python_chunks_fallback_mode(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text(
        "def bad(\n"
        "    return 1\n"
        "\n"
        "x = 2\n",
        encoding="utf-8",
    )

    records = extract_python_chunks(broken, root=tmp_path, encoder=FakeEncoder())

    assert records
    assert all(item["symbol_type"] == "line_chunk" for item in records)
    assert all(item["ast_parse_ok"] is False for item in records)


def test_build_chunks_scans_directory(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "b.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    records = build_chunks(tmp_path, encoder=FakeEncoder())

    source_ids = {item["source_id"] for item in records}
    assert "a.py" in source_ids
    assert "pkg/b.py" in source_ids
