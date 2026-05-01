from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from tokenizer_utils import count_gpt4o_base_tokens


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    symbol_type: str
    symbol_name: str
    start_line: int
    end_line: int
    ast_parse_ok: bool
    raw_tokens: int
    chunk_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_python_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        if input_path.suffix == ".py":
            yield input_path
        return
    for path in sorted(input_path.rglob("*.py")):
        if path.is_file():
            yield path


def build_chunks(input_path: Path, *, encoder: Any | None = None) -> list[dict[str, Any]]:
    root = input_path if input_path.is_dir() else input_path.parent
    records: list[dict[str, Any]] = []
    for py_file in iter_python_files(input_path):
        records.extend(extract_python_chunks(py_file, root=root, encoder=encoder))
    return records


def extract_python_chunks(
    path: Path,
    *,
    root: Path,
    encoder: Any | None = None,
    fallback_max_lines: int = 200,
) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    source_id = path.relative_to(root).as_posix()
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            record.to_dict()
            for record in _fallback_line_chunks(
                source_id=source_id,
                source=source,
                lines=lines,
                encoder=encoder,
                max_lines=fallback_max_lines,
            )
        ]

    records: list[ChunkRecord] = []
    total_lines = max(1, len(lines))
    records.append(
        _make_record(
            source_id=source_id,
            symbol_type="module",
            symbol_name=path.stem,
            start_line=1,
            end_line=total_lines,
            lines=lines,
            ast_parse_ok=True,
            encoder=encoder,
        )
    )

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._record("class", node.name, node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record("function", node.name, node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record("async_function", node.name, node)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def _record(self, symbol_type: str, local_name: str, node: ast.AST) -> None:
            start_line = int(getattr(node, "lineno", 1))
            end_line = int(getattr(node, "end_lineno", start_line))
            qualified = ".".join([*self.scope, local_name]) if self.scope else local_name
            records.append(
                _make_record(
                    source_id=source_id,
                    symbol_type=symbol_type,
                    symbol_name=qualified,
                    start_line=start_line,
                    end_line=end_line,
                    lines=lines,
                    ast_parse_ok=True,
                    encoder=encoder,
                )
            )

    Visitor().visit(tree)
    return [record.to_dict() for record in records]


def _make_record(
    *,
    source_id: str,
    symbol_type: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
    lines: list[str],
    ast_parse_ok: bool,
    encoder: Any | None,
) -> ChunkRecord:
    text = "\n".join(lines[start_line - 1 : end_line])
    chunk_id = f"{source_id}::{symbol_type}::{symbol_name}::{start_line}-{end_line}"
    raw_tokens = count_gpt4o_base_tokens(text, encoder=encoder)
    return ChunkRecord(
        chunk_id=chunk_id,
        source_id=source_id,
        symbol_type=symbol_type,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        ast_parse_ok=ast_parse_ok,
        raw_tokens=raw_tokens,
        chunk_text=text,
    )


def _fallback_line_chunks(
    *,
    source_id: str,
    source: str,
    lines: list[str],
    encoder: Any | None,
    max_lines: int,
) -> list[ChunkRecord]:
    if not lines:
        return [
            ChunkRecord(
                chunk_id=f"{source_id}::line_chunk::0::1-1",
                source_id=source_id,
                symbol_type="line_chunk",
                symbol_name="fallback_0",
                start_line=1,
                end_line=1,
                ast_parse_ok=False,
                raw_tokens=count_gpt4o_base_tokens(source, encoder=encoder),
                chunk_text=source,
            )
        ]

    ranges: list[tuple[int, int]] = []
    start = 1
    total = len(lines)
    while start <= total:
        end = min(total, start + max_lines - 1)
        if end < total:
            blank = _find_last_blank_line(lines, start, end)
            if blank is not None and blank >= start:
                end = blank
        if end < start:
            end = min(total, start + max_lines - 1)
        ranges.append((start, end))
        start = end + 1

    return [
        _make_record(
            source_id=source_id,
            symbol_type="line_chunk",
            symbol_name=f"fallback_{index}",
            start_line=start_line,
            end_line=end_line,
            lines=lines,
            ast_parse_ok=False,
            encoder=encoder,
        )
        for index, (start_line, end_line) in enumerate(ranges)
    ]


def _find_last_blank_line(lines: list[str], start_line: int, end_line: int) -> int | None:
    for idx in range(end_line, start_line - 1, -1):
        if not lines[idx - 1].strip():
            return idx
    return None
