"""Stage 2: optional R01–R05 (comments, blanks, ws, docstrings, indent). R04/R05 are lossy."""

import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CleaningConfig:
    remove_comments:            bool = False
    remove_blank_lines:         bool = True
    remove_trailing_whitespace: bool = True
    remove_docstrings:          bool = False
    """When True, only **low-risk** docstrings are removed (AST-based, see ``docstring_removal_mode``)."""
    docstring_removal_mode:     str = "safe_only"
    """``off`` | ``safe_only``. ``remove_docstrings=True`` implies ``safe_only`` (conservative)."""
    remove_indentation:         bool = True
    b_starvation_probe:       bool = False
    """Opt-in: retain a thin slice of long, space-rich docstrings for Stage3 B funnel experiments."""
    b_probe_min_inner_chars:    int = 64
    b_probe_min_space_ratio:  float = 0.12


def _merge_docstring_reports(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not a:
        return b
    if not b:
        return a
    out = {
        "removed_count": int(a.get("removed_count", 0)) + int(b.get("removed_count", 0)),
        "kept_count": int(a.get("kept_count", 0)) + int(b.get("kept_count", 0)),
        "removed": list(a.get("removed", [])) + list(b.get("removed", [])),
        "kept": list(a.get("kept", [])) + list(b.get("kept", [])),
        "parse_failed": bool(a.get("parse_failed")) or bool(b.get("parse_failed")),
        "b_probe_retained_count": int(a.get("b_probe_retained_count", 0))
        + int(b.get("b_probe_retained_count", 0)),
        "b_probe_retained_chars": int(a.get("b_probe_retained_chars", 0))
        + int(b.get("b_probe_retained_chars", 0)),
    }
    pa, pb = a.get("path_context"), b.get("path_context")
    out["path_context"] = pa if pa else pb
    return out


@dataclass
class CleaningStats:
    original_chars:             int = 0
    cleaned_chars:             int = 0
    removed_blank_lines:        int = 0
    removed_docstring_chars:    int = 0
    removed_indent_chars:       int = 0
    docstring_removal_report:   dict[str, Any] | None = None
    stage2_removed_comment_count: int = 0
    stage2_removed_comment_tokens_est: int = 0
    stage2_removed_docstring_count: int = 0
    stage2_removed_docstring_tokens_est: int = 0
    stage2_removed_free_text_estimated_tokens: int = 0
    stage2_retained_for_b_probe_count: int = 0
    stage2_retained_for_b_probe_tokens_est: int = 0

    @property
    def char_reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return (1.0 - self.cleaned_chars / self.original_chars) * 100.0

    def __add__(self, other: "CleaningStats") -> "CleaningStats":
        return CleaningStats(
            original_chars=self.original_chars + other.original_chars,
            cleaned_chars=self.cleaned_chars + other.cleaned_chars,
            removed_blank_lines=self.removed_blank_lines + other.removed_blank_lines,
            removed_docstring_chars=self.removed_docstring_chars + other.removed_docstring_chars,
            removed_indent_chars=self.removed_indent_chars + other.removed_indent_chars,
            docstring_removal_report=_merge_docstring_reports(
                self.docstring_removal_report,
                other.docstring_removal_report,
            ),
            stage2_removed_comment_count=self.stage2_removed_comment_count
            + other.stage2_removed_comment_count,
            stage2_removed_comment_tokens_est=self.stage2_removed_comment_tokens_est
            + other.stage2_removed_comment_tokens_est,
            stage2_removed_docstring_count=self.stage2_removed_docstring_count
            + other.stage2_removed_docstring_count,
            stage2_removed_docstring_tokens_est=self.stage2_removed_docstring_tokens_est
            + other.stage2_removed_docstring_tokens_est,
            stage2_removed_free_text_estimated_tokens=self.stage2_removed_free_text_estimated_tokens
            + other.stage2_removed_free_text_estimated_tokens,
            stage2_retained_for_b_probe_count=self.stage2_retained_for_b_probe_count
            + other.stage2_retained_for_b_probe_count,
            stage2_retained_for_b_probe_tokens_est=self.stage2_retained_for_b_probe_tokens_est
            + other.stage2_retained_for_b_probe_tokens_est,
        )


_CODING_COOKIE_RE = re.compile(
    r"^[ \t\f]*#.*?coding[:=]\s*([-\w.]+)",
    re.IGNORECASE,
)


def is_preserved_directive_comment(
    comment_text: str,
    line_no: int,
    *,
    source_line: str,
) -> bool:
    """
    *comment_text*: payload after ``#`` (may include leading space).
    *source_line*: full physical line (for shebang / encoding).
    """
    sl = source_line.rstrip("\n\r")
    st = sl.lstrip()
    if line_no == 1 and st.startswith("#!"):
        return True
    if line_no <= 2 and _CODING_COOKIE_RE.match(sl):
        return True

    body = comment_text.strip()
    low = body.lower()
    if low.startswith("type:") and "ignore" in low:
        return True
    if low.startswith("noqa") or low.startswith("noqa:"):
        return True
    if "pragma:" in low and "no cover" in low:
        return True
    if low.startswith("pylint:") or low.startswith("pylint disable"):
        return True
    if low.startswith("fmt: off") or low.startswith("fmt: on"):
        return True
    if low.startswith("pyright:") or low.startswith("mypy:"):
        return True
    return False


def find_multiline_string_line_spans(source: str) -> set[int]:
    """1-based line numbers that lie inside a *multi-line* string token."""
    protected: set[int] = set()
    try:
        readline = io.StringIO(source).readline
        toks = list(tokenize.generate_tokens(readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return protected
    for tok in toks:
        if tok.type != tokenize.STRING:
            continue
        sline, _ = tok.start
        eline, _ = tok.end
        if eline > sline:
            for ln in range(sline, eline + 1):
                protected.add(ln)
    return protected


def _remove_comments(
    source: str, *, preserve_directives: bool = True
) -> tuple[str, int, int]:
    """Remove ``#`` comments; returns (text, removed_line_count, est_tokens).

    Token estimate: ``max(1, len(comment_token)//4)`` per removed COMMENT token
    (``tokenize``), or for regex fallback ``max(1, len(match)//4)`` per line.
    """
    lines = source.splitlines(keepends=True)
    if not preserve_directives:
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            out = re.sub(r"(?m)#[^\n]*", "", source)
            n_lines = len(re.findall(r"(?m)#", source))
            est = 0
            for m in re.finditer(r"(?m)#([^\n]*)", source):
                est += max(1, len(m.group(0)) // 4)
            return out, n_lines, est

        comment_cols: dict[int, int] = {}
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                comment_cols[tok.start[0]] = tok.start[1]

        if not comment_cols:
            return source, 0, 0

        result = []
        rm_lines = 0
        est_tok = 0
        for i, line in enumerate(lines, start=1):
            if i in comment_cols:
                rm_lines += 1
                col = comment_cols[i]
                stripped = line[:col].rstrip()
                est_tok += max(1, (len(line) - col) // 4)
                result.append(stripped + "\n" if stripped else "\n")
            else:
                result.append(line)
        return "".join(result), rm_lines, est_tok

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out = re.sub(r"(?m)#[^\n]*", "", source)
        est = 0
        for m in re.finditer(r"(?m)#([^\n]*)", source):
            est += max(1, len(m.group(0)) // 4)
        return out, len(re.findall(r"(?m)#", source)), est

    comment_cols: dict[int, int] = {}
    comment_tok_len: dict[int, int] = {}
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        raw_line = lines[row - 1] if 0 < row <= len(lines) else ""
        payload = tok.string[1:] if tok.string.startswith("#") else tok.string
        if is_preserved_directive_comment(
            payload, row, source_line=raw_line.split("\n", 1)[0]
        ):
            continue
        comment_cols[row] = col
        comment_tok_len[row] = max(1, len(tok.string) // 4)

    if not comment_cols:
        return source, 0, 0

    result = []
    rm_lines = 0
    est_tok = 0
    for i, line in enumerate(lines, start=1):
        if i in comment_cols:
            rm_lines += 1
            col = comment_cols[i]
            stripped = line[:col].rstrip()
            est_tok += int(comment_tok_len.get(i, max(1, (len(line) - col) // 4)))
            result.append(stripped + "\n" if stripped else "\n")
        else:
            result.append(line)
    return "".join(result), rm_lines, est_tok


def clean_code(
    source: str,
    config: Optional[CleaningConfig] = None,
    *,
    path: str | None = None,
) -> tuple[str, CleaningStats]:
    """R05 → R01 → R03 → R02 → R04 when each flag is on."""
    if config is None:
        config = CleaningConfig()

    stats = CleaningStats(original_chars=len(source))

    if config.remove_docstrings:
        mode = getattr(config, "docstring_removal_mode", "safe_only")
        if mode == "off":
            pass
        elif mode == "safe_only":
            from stage2.docstring_analysis import DocstringInfo, remove_safe_docstrings

            before = len(source)

            def _retain_doc_for_b_probe(info: DocstringInfo) -> bool:
                inner = info.text.strip()
                if len(inner) < int(config.b_probe_min_inner_chars):
                    return False
                sp = inner.count(" ") + inner.count("\n") + inner.count("\t")
                ratio = sp / max(len(inner), 1)
                return ratio >= float(config.b_probe_min_space_ratio)

            source, report = remove_safe_docstrings(
                source,
                path=path,
                retain_if=_retain_doc_for_b_probe if config.b_starvation_probe else None,
            )
            stats.removed_docstring_chars = before - len(source)
            stats.docstring_removal_report = report
            stats.stage2_removed_docstring_count = int(report.get("removed_count", 0))
            for row in report.get("removed", []) or []:
                ic = int(row.get("inner_chars", 0))
                stats.stage2_removed_docstring_tokens_est += max(1, ic // 4)
                if row.get("routes_b_candidate"):
                    stats.stage2_removed_free_text_estimated_tokens += max(1, ic // 4)
            prc = int(report.get("b_probe_retained_chars", 0))
            stats.stage2_retained_for_b_probe_count = int(report.get("b_probe_retained_count", 0))
            stats.stage2_retained_for_b_probe_tokens_est = max(1, prc // 4) if prc else 0
        else:
            raise ValueError(f"unknown docstring_removal_mode: {mode!r}")

    if config.remove_comments:
        source, cc, ctk = _remove_comments(source, preserve_directives=True)
        stats.stage2_removed_comment_count += cc
        stats.stage2_removed_comment_tokens_est += ctk

    protected: set[int] = set()
    if (
        config.remove_trailing_whitespace
        or config.remove_blank_lines
        or config.remove_indentation
    ):
        protected = find_multiline_string_line_spans(source)

    lines = source.splitlines()

    if config.remove_trailing_whitespace:
        lines = [
            ln if (i + 1) in protected else ln.rstrip()
            for i, ln in enumerate(lines)
        ]

    if config.remove_blank_lines:
        original_count = len(lines)
        new_lines: list[str] = []
        for i, ln in enumerate(lines):
            if (i + 1) in protected:
                new_lines.append(ln)
            elif ln.strip():
                new_lines.append(ln)
        stats.removed_blank_lines = original_count - len(new_lines)
        lines = new_lines

    if config.remove_indentation:
        indent_chars = 0
        out_ind: list[str] = []
        for i, ln in enumerate(lines):
            if (i + 1) in protected:
                out_ind.append(ln)
            else:
                indent_chars += len(ln) - len(ln.lstrip())
                out_ind.append(ln.lstrip())
        stats.removed_indent_chars = indent_chars
        lines = out_ind

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
