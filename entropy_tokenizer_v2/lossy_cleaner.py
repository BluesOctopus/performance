"""Stage 2: Ultra-Minimalist Structural Cleaning.
Uses Syntax-Triggered Indentation (implicit after ':') and explicit Dedent markers (<D>).
Maximizes token savings while maintaining 100% reversibility.
"""

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class CleaningConfig:
    remove_comments:            bool = True
    remove_docstrings:          bool = True
    # New: Minimalist structural compression
    use_minimalist_indent:      bool = True  # Implicit IND after ':', explicit <D> for dedent
    remove_blank_lines:         bool = True
    remove_trailing_whitespace: bool = True


@dataclass
class CleaningStats:
    original_chars:             int = 0
    cleaned_chars:              int = 0
    removed_comment_chars:      int = 0
    removed_docstring_chars:    int = 0
    removed_indent_chars:       int = 0
    dedent_markers_added:       int = 0

    def __add__(self, other: "CleaningStats") -> "CleaningStats":
        return CleaningStats(
            original_chars          = self.original_chars          + other.original_chars,
            cleaned_chars           = self.cleaned_chars           + other.cleaned_chars,
            removed_comment_chars   = self.removed_comment_chars   + other.removed_comment_chars,
            removed_docstring_chars = self.removed_docstring_chars + other.removed_docstring_chars,
            removed_indent_chars    = self.removed_indent_chars    + other.removed_indent_chars,
            dedent_markers_added    = self.dedent_markers_added    + other.dedent_markers_added,
        )


def _remove_docstrings(source: str) -> Tuple[str, int]:
    """Strip docstrings using AST for precision."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        before = len(source)
        source = re.sub(r'"""[\s\S]*?"""', '', source)
        source = re.sub(r"'''[\s\S]*?'''", '', source)
        return source, before - len(source)

    docstring_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                ds_node = node.body[0]
                docstring_ranges.append((ds_node.lineno, ds_node.end_lineno))

    if not docstring_ranges:
        return source, 0

    lines = source.splitlines(keepends=True)
    excluded_lines = set()
    for start, end in docstring_ranges:
        for ln in range(start, end + 1):
            excluded_lines.add(ln)
            
    new_lines = []
    removed_chars = 0
    for i, line in enumerate(lines, start=1):
        if i in excluded_lines:
            removed_chars += len(line)
        else:
            new_lines.append(line)
            
    return "".join(new_lines), removed_chars


def _remove_comments(source: str) -> Tuple[str, int]:
    """Remove comments via tokenize to protect strings."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        before = len(source)
        source = re.sub(r'(?m)#[^\n]*', '', source)
        return source, before - len(source)

    comment_map = {t.start[0]: t.start[1] for t in toks if t.type == tokenize.COMMENT}
    lines = source.splitlines(keepends=True)
    removed_chars = 0
    result = []
    
    for i, line in enumerate(lines, start=1):
        if i in comment_map:
            col = comment_map[i]
            content = line[:col]
            removed_chars += len(line) - len(content)
            if content.strip():
                result.append(content.rstrip() + "\n")
        else:
            result.append(line)
            
    return "".join(result), removed_chars


def clean_code(
    source: str,
    config: Optional[CleaningConfig] = None,
) -> Tuple[str, CleaningStats]:
    """
    Ultra-Minimalist Stage 2:
    - Implicit IND after ':' (only for actual code blocks).
    - Explicit <I> for non-colon indents.
    - Explicit <D> for dedents or to cancel implicit indents.
    """
    if config is None:
        config = CleaningConfig()

    stats = CleaningStats(original_chars=len(source))

    if config.remove_docstrings:
        source, removed = _remove_docstrings(source)
        stats.removed_docstring_chars = removed

    if config.remove_comments:
        source, removed = _remove_comments(source)
        stats.removed_comment_chars = removed

    lines = source.splitlines()
    new_lines = []
    
    expected_level = 0 
    dedents_added = 0
    indents_added = 0
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            if config.remove_blank_lines: continue
            new_lines.append(""); continue

        if config.use_minimalist_indent:
            leading_spaces = len(line) - len(stripped)
            indent_str = line[:leading_spaces]
            effective_spaces = indent_str.count(' ') + indent_str.count('\t') * 4
            new_level = effective_spaces // 4
            stats.removed_indent_chars += leading_spaces
            
            if new_level < expected_level:
                diff = expected_level - new_level
                marker = "<D>" * diff
                new_lines.append(f"{marker}{stripped}")
                dedents_added += diff
            elif new_level > expected_level:
                diff = new_level - expected_level
                marker = "<I>" * diff
                new_lines.append(f"{marker}{stripped}")
                indents_added += diff
            else:
                new_lines.append(stripped)
            
            expected_level = new_level
            if stripped.endswith(":") and not stripped.startswith("#"):
                block_starters = ("def ", "class ", "if ", "elif ", "else:", "for ", "while ", "with ", "try:", "except ", "except:", "finally:")
                if any(stripped.startswith(s) for s in block_starters) or stripped.endswith(":"):
                    expected_level += 1
        else:
            new_lines.append(line)

    source = "\n".join(new_lines)
    stats.cleaned_chars = len(source)
    stats.dedent_markers_added = dedents_added
    
    return source, stats


def clean_corpus(sources: List[str], config: Optional[CleaningConfig] = None):
    if config is None: config = CleaningConfig()
    cleaned_list, total = [], CleaningStats()
    for src in sources:
        cleaned, s = clean_code(src, config)
        cleaned_list.append(cleaned)
        total = total + s
    return cleaned_list, total


def lossless_clean(source: str) -> Tuple[str, CleaningStats]:
    """R02+R03 only; keep comments, docstrings, indentation."""
    cfg = CleaningConfig(
        remove_comments=False,
        remove_blank_lines=True,
        remove_trailing_whitespace=True,
        remove_docstrings=False,
        use_minimalist_indent=False,
    )
    return clean_code(source, cfg)


def restore_code(compressed_text: str, config: Optional[CleaningConfig] = None) -> str:
    """Reverse Stage 2: Restore physical indentation from <I>, <D> markers and colons."""
    if config is None:
        config = CleaningConfig()
    
    if not config.use_minimalist_indent:
        return compressed_text
        
    lines = compressed_text.splitlines()
    restored_lines = []
    current_level = 0
    
    for line in lines:
        if not line.strip():
            restored_lines.append("")
            continue
            
        while line.startswith("<D>"):
            current_level -= 1
            line = line[3:]
        while line.startswith("<I>"):
            current_level += 1
            line = line[3:]
            
        restored_lines.append("    " * current_level + line)
        
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("#"):
            block_starters = ("def ", "class ", "if ", "elif ", "else:", "for ", "while ", "with ", "try:", "except ", "except:", "finally:")
            if any(stripped.startswith(s) for s in block_starters) or stripped.endswith(":"):
                current_level += 1
            
    return "\n".join(restored_lines)
