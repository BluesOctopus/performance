"""Stage 2: optional R01–R05 (comments, blanks, ws, docstrings, indent). R04/R05 are lossy."""

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CleaningConfig:
    remove_comments:            bool = False
    remove_blank_lines:         bool = True
    remove_trailing_whitespace: bool = True
    remove_docstrings:          bool = False
    remove_indentation:         bool = True


@dataclass
class CleaningStats:
    original_chars:             int = 0
    cleaned_chars:              int = 0
    removed_blank_lines:        int = 0
    removed_docstring_chars:    int = 0
    removed_indent_chars:       int = 0

    @property
    def char_reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return (1.0 - self.cleaned_chars / self.original_chars) * 100.0

    def __add__(self, other: "CleaningStats") -> "CleaningStats":
        return CleaningStats(
            original_chars          = self.original_chars          + other.original_chars,
            cleaned_chars           = self.cleaned_chars           + other.cleaned_chars,
            removed_blank_lines     = self.removed_blank_lines     + other.removed_blank_lines,
            removed_docstring_chars = self.removed_docstring_chars + other.removed_docstring_chars,
            removed_indent_chars    = self.removed_indent_chars    + other.removed_indent_chars,
        )


def _remove_docstrings(source: str) -> tuple[str, int]:
    """Strip first docstring in module/class/function bodies; AST first, else regex."""
    removed_chars = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        before = len(source)
        source = re.sub(r'"""[\s\S]*?"""', '', source)
        source = re.sub(r"'''[\s\S]*?'''", '', source)
        return source, before - len(source)

    docstring_line_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Module)):
            if (node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                ds = node.body[0]
                docstring_line_ranges.append((ds.lineno, ds.end_lineno))

    if not docstring_line_ranges:
        return source, 0

    excluded: set[int] = set()
    for start, end in docstring_line_ranges:
        for ln in range(start, end + 1):
            excluded.add(ln)

    lines = source.splitlines(keepends=True)
    kept, removed = [], []
    for i, line in enumerate(lines, start=1):
        if i in excluded:
            removed.append(line)
        else:
            kept.append(line)

    removed_chars = sum(len(l) for l in removed)
    return "".join(kept), removed_chars


def _remove_comments(source: str) -> str:
    """Remove ``#`` comments via ``tokenize`` (strings safe); regex fallback."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.sub(r'(?m)#[^\n]*', '', source)

    comment_cols: dict[int, int] = {}
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            comment_cols[tok.start[0]] = tok.start[1]

    if not comment_cols:
        return source

    result = []
    for i, line in enumerate(source.splitlines(keepends=True), start=1):
        if i in comment_cols:
            col = comment_cols[i]
            stripped = line[:col].rstrip()
            result.append(stripped + "\n" if stripped else "\n")
        else:
            result.append(line)
    return "".join(result)


def clean_code(
    source: str,
    config: Optional[CleaningConfig] = None,
) -> tuple[str, CleaningStats]:
    """R05 → R01 → R03 → R02 → R04 when each flag is on."""
    if config is None:
        config = CleaningConfig()

    stats = CleaningStats(original_chars=len(source))

    if config.remove_docstrings:
        source, removed = _remove_docstrings(source)
        stats.removed_docstring_chars = removed

    if config.remove_comments:
        source = _remove_comments(source)

    lines = source.splitlines()

    if config.remove_trailing_whitespace:
        lines = [ln.rstrip() for ln in lines]

    if config.remove_blank_lines:
        original_count = len(lines)
        lines = [ln for ln in lines if ln.strip()]
        stats.removed_blank_lines = original_count - len(lines)

    if config.remove_indentation:
        indent_chars = sum(len(ln) - len(ln.lstrip()) for ln in lines)
        stats.removed_indent_chars = indent_chars
        lines = [ln.lstrip() for ln in lines]

    source = "\n".join(lines)
    stats.cleaned_chars = len(source)
    return source, stats


def clean_corpus(
    sources: list[str],
    config: Optional[CleaningConfig] = None,
) -> tuple[list[str], CleaningStats]:
    """Clean a list of code strings and return aggregate stats."""
    if config is None:
        config = CleaningConfig()

    cleaned_list: list[str] = []
    total = CleaningStats()
    for src in sources:
        cleaned, s = clean_code(src, config)
        cleaned_list.append(cleaned)
        total = total + s
    return cleaned_list, total


def lossless_clean(source: str) -> tuple[str, CleaningStats]:
    """R02+R03 only; keep comments, docstrings, indentation."""
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=True,
        remove_trailing_whitespace=True,
        remove_docstrings=False,
        remove_indentation=False,
    )
    return clean_code(source, cfg)


def lossy_clean(source: str) -> tuple[str, CleaningStats]:
    """``CleaningConfig()`` defaults (R04 on; comments/docstrings off)."""
    return clean_code(source, CleaningConfig())
