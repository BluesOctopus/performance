"""Conservative AST-only docstring analysis and safe removal (Stage2)."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def normalize_path_for_docstring_policy(path: str | None) -> str:
    """Lowercase path with forward slashes; empty string if *path* is None."""
    if not path:
        return ""
    return path.replace("\\", "/").lower()


def classify_docstring_path_context(path: str | None) -> dict[str, Any]:
    """
    Classify filesystem path for conservative docstring policy hints.

    ``is_low_risk_context`` is True if any test/script/example/internal heuristic matches.
    """
    norm = normalize_path_for_docstring_policy(path)
    labels: list[str] = []
    base = os.path.basename(norm) if norm else ""
    # Slash-padded so ``proj/scripts/x.py`` and ``scripts/x.py`` both match ``/scripts/``.
    padded = f"/{norm}/" if norm else ""

    is_test_file = (
        "/tests/" in padded
        or "/test/" in padded
        or base.startswith("test_")
        or base.endswith("_test.py")
    )
    if is_test_file:
        labels.append("tests")

    is_script_file = "/scripts/" in padded or "/bin/" in padded
    if is_script_file:
        labels.append("scripts")

    is_example_file = (
        "/examples/" in padded
        or "/example/" in padded
        or "/demo/" in padded
    )
    if is_example_file:
        labels.append("examples")

    is_internal_file = (
        "/internal/" in padded
        or "/utils/internal/" in padded
        or "/private/" in padded
    )
    if is_internal_file:
        labels.append("internal")

    is_low_risk_context = bool(labels)

    return {
        "normalized_path": norm,
        "is_test_file": is_test_file,
        "is_script_file": is_script_file,
        "is_example_file": is_example_file,
        "is_internal_file": is_internal_file,
        "is_low_risk_context": is_low_risk_context,
        "matched_context_labels": labels,
    }


def is_low_risk_nested_docstring(info: DocstringInfo) -> bool:
    """
    True when docstring is on a nested callable/class with no extra risk signals.

    Not sufficient alone for removal; combined with ``decide_docstring_removal``.
    """
    if not info.is_nested:
        return False
    if info.kind not in ("function", "method", "class"):
        return False
    if info.has_structured_markers:
        return False
    if has_high_risk_decorators(info.decorators):
        return False
    if info.name in ("__init__", "__call__"):
        return False
    if re.fullmatch(r"__\w+__", info.name):
        return False
    return True


def decide_docstring_removal(
    info: DocstringInfo,
    *,
    path_context: dict[str, Any],
    usage_signals: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Layered removal decision. Returns ``(remove_candidate, risk_reasons)``.
    """
    # --- Layer 1: file-level runtime doc usage ---
    if usage_signals.get("any_signal"):
        return False, [
            "file_has_docstring_runtime_usage_signal:"
            + ",".join(usage_signals.get("raw_hits") or []),
        ]

    # --- Layer 2: always preserve (module-level / API / risk signals) ---
    layer2: list[str] = []

    if info.kind == "module":
        layer2.append("module_docstring_always_preserved")

    if info.kind == "class" and info.is_public and not info.is_nested:
        layer2.append("public_class_docstring_preserved")

    if info.kind in ("function", "method") and info.is_public and not info.is_nested:
        layer2.append("public_function_or_method_docstring_preserved")

    if info.name in ("__init__", "__call__"):
        layer2.append("special_method_init_or_call_preserved")

    if re.fullmatch(r"__\w+__", info.name):
        layer2.append("dunder_method_docstring_preserved")

    if info.has_structured_markers:
        layer2.append("structured_doc_markers_present")

    if has_high_risk_decorators(info.decorators):
        layer2.append("high_risk_decorator:" + ",".join(info.decorators))

    if layer2:
        return False, layer2

    # --- Layer 3: low-risk removable ---
    reasons: list[str] = []
    can_remove = False

    if info.kind in ("function", "method") and info.is_private:
        can_remove = True
        reasons.append("low_risk_private_docstring_eligible_for_removal")
    elif info.kind == "class" and info.is_private:
        can_remove = True
        reasons.append("low_risk_private_class_docstring_eligible")
    elif is_low_risk_nested_docstring(info):
        if info.is_public and path_context.get("is_low_risk_context"):
            return False, ["public_nested_docstring_preserved_in_low_risk_path_context"]
        can_remove = True
        reasons.append("nested_low_risk_docstring_eligible")

    if can_remove and path_context.get("is_low_risk_context"):
        labels = path_context.get("matched_context_labels") or []
        reasons.append("low_risk_path_context:" + ",".join(labels))

    if can_remove:
        return True, reasons

    # --- Layer 4 ---
    return False, ["uncertain_default_preserve"]


@dataclass
class DocstringInfo:
    kind: str
    name: str
    qualname: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    text: str
    is_public: bool
    is_private: bool
    is_nested: bool
    decorators: list[str]
    has_structured_markers: bool
    risk_reasons: list[str] = field(default_factory=list)
    remove_candidate: bool = False


def _is_docstring_expr(stmt: ast.stmt | None) -> ast.Expr | None:
    if stmt is None:
        return None
    if not isinstance(stmt, ast.Expr):
        return None
    v = stmt.value
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return stmt
    return None


def extract_decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    out: list[str] = []
    for d in node.decorator_list:
        try:
            out.append(ast.unparse(d).strip().replace("\n", " "))
        except Exception:
            if isinstance(d, ast.Name):
                out.append(d.id)
            elif isinstance(d, ast.Attribute):
                parts: list[str] = []
                cur: ast.expr = d
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                out.append(".".join(reversed(parts)))
    return out


def has_high_risk_decorators(decorators: list[str]) -> bool:
    joined = " ".join(decorators).lower()
    risks = (
        "property",
        "cached_property",
        ".get",
        ".post",
        ".put",
        ".delete",
        ".patch",
        "click.command",
        "click.group",
        "validator",
        "field_validator",
        "root_validator",
        "computed_field",
    )
    return any(r in joined for r in risks)


_STRUCTURED_PATTERNS = re.compile(
    r"(?is)"
    r"\bargs\s*:\s*|"
    r"\breturns\s*:\s*|"
    r"\braises\s*:\s*|"
    r"\bparameters\b|"
    r"\bexamples\b|"
    r"\byields\s*:\s*|"
    r":param\b|"
    r":return\s*:|"
    r"@param\b|"
    r"@return\b"
)


def has_structured_doc_markers(text: str) -> bool:
    return bool(_STRUCTURED_PATTERNS.search(text))


def find_docstring_usage_signals(source: str) -> dict[str, Any]:
    raw_hits: list[str] = []
    uses___doc__ = bool(re.search(r"\b__doc__\b", source))
    if uses___doc__:
        raw_hits.append("__doc__")
    uses_inspect_getdoc = bool(re.search(r"inspect\s*\.\s*getdoc\s*\(", source))
    if uses_inspect_getdoc:
        raw_hits.append("inspect.getdoc")
    uses_inspect_cleandoc = bool(re.search(r"inspect\s*\.\s*cleandoc\s*\(", source))
    if uses_inspect_cleandoc:
        raw_hits.append("inspect.cleandoc")
    uses_help = bool(re.search(r"\bhelp\s*\(", source))
    if uses_help:
        raw_hits.append("help(")
    uses_pydoc = bool(re.search(r"\bpydoc\b", source))
    if uses_pydoc:
        raw_hits.append("pydoc")
    uses_ast_get_docstring = bool(re.search(r"ast\s*\.\s*get_docstring\s*\(", source))
    if uses_ast_get_docstring:
        raw_hits.append("ast.get_docstring")

    return {
        "uses___doc__": uses___doc__,
        "uses_inspect_getdoc": uses_inspect_getdoc,
        "uses_inspect_cleandoc": uses_inspect_cleandoc,
        "uses_help": uses_help,
        "uses_pydoc": uses_pydoc,
        "uses_ast_get_docstring": uses_ast_get_docstring,
        "raw_hits": raw_hits,
        "any_signal": bool(raw_hits),
    }


def iter_docstrings(source: str) -> list[DocstringInfo]:
    tree = ast.parse(source)
    results: list[DocstringInfo] = []

    # Stack: (kind, name, qualname, in_function_depth)
    stack: list[tuple[str, str, str]] = []

    def function_depth() -> int:
        return sum(1 for t in stack if t[0] in ("function", "async_function"))

    def current_qualname() -> str:
        parts = [t[2] for t in stack if t[2]]
        return ".".join(parts) if parts else "<module>"

    def visit_function_like(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
    ) -> None:
        nonlocal results
        qn = current_qualname()
        my_qn = f"{qn}.{node.name}" if qn != "<module>" else node.name
        decorators = extract_decorator_names(node)
        ds_expr = _is_docstring_expr(node.body[0] if node.body else None)
        nested = function_depth() > 0
        is_private = node.name.startswith("_") and not (
            node.name.startswith("__") and node.name.endswith("__")
        )
        is_public = not is_private

        if ds_expr:
            val = ds_expr.value
            text = val.value if isinstance(val, ast.Constant) else ""
            assert isinstance(ds_expr, ast.Expr)
            results.append(
                DocstringInfo(
                    kind="method" if stack and stack[-1][0] == "class" else "function",
                    name=node.name,
                    qualname=my_qn,
                    lineno=ds_expr.lineno,
                    end_lineno=ds_expr.end_lineno or ds_expr.lineno,
                    col_offset=ds_expr.col_offset,
                    end_col_offset=ds_expr.end_col_offset or ds_expr.col_offset,
                    text=text,
                    is_public=is_public,
                    is_private=is_private,
                    is_nested=nested,
                    decorators=decorators,
                    has_structured_markers=has_structured_doc_markers(text),
                )
            )

        stack.append((kind, node.name, my_qn))
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_function_like(
                    child,
                    "async_function" if isinstance(child, ast.AsyncFunctionDef) else "function",
                )
            elif isinstance(child, ast.ClassDef):
                visit_class(child)
        stack.pop()

    def visit_class(node: ast.ClassDef) -> None:
        nonlocal results
        qn = current_qualname()
        my_qn = f"{qn}.{node.name}" if qn != "<module>" else node.name
        decorators = extract_decorator_names(node)
        ds_expr = _is_docstring_expr(node.body[0] if node.body else None)
        is_private = node.name.startswith("_") and not (
            node.name.startswith("__") and node.name.endswith("__")
        )
        is_public = not is_private

        if ds_expr:
            val = ds_expr.value
            text = val.value if isinstance(val, ast.Constant) else ""
            assert isinstance(ds_expr, ast.Expr)
            results.append(
                DocstringInfo(
                    kind="class",
                    name=node.name,
                    qualname=my_qn,
                    lineno=ds_expr.lineno,
                    end_lineno=ds_expr.end_lineno or ds_expr.lineno,
                    col_offset=ds_expr.col_offset,
                    end_col_offset=ds_expr.end_col_offset or ds_expr.col_offset,
                    text=text,
                    is_public=is_public,
                    is_private=is_private,
                    is_nested=function_depth() > 0,
                    decorators=decorators,
                    has_structured_markers=has_structured_doc_markers(text),
                )
            )

        stack.append(("class", node.name, my_qn))
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                visit_function_like(child, "function")
            elif isinstance(child, ast.AsyncFunctionDef):
                visit_function_like(child, "async_function")
            elif isinstance(child, ast.ClassDef):
                visit_class(child)
        stack.pop()

    # module docstring
    if tree.body:
        ds_expr = _is_docstring_expr(tree.body[0])
        if ds_expr:
            val = ds_expr.value
            text = val.value if isinstance(val, ast.Constant) else ""
            assert isinstance(ds_expr, ast.Expr)
            results.append(
                DocstringInfo(
                    kind="module",
                    name="<module>",
                    qualname="<module>",
                    lineno=ds_expr.lineno,
                    end_lineno=ds_expr.end_lineno or ds_expr.lineno,
                    col_offset=ds_expr.col_offset,
                    end_col_offset=ds_expr.end_col_offset or ds_expr.col_offset,
                    text=text,
                    is_public=True,
                    is_private=False,
                    is_nested=False,
                    decorators=[],
                    has_structured_markers=has_structured_doc_markers(text),
                )
            )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            visit_class(node)
        elif isinstance(node, ast.FunctionDef):
            visit_function_like(node, "function")
        elif isinstance(node, ast.AsyncFunctionDef):
            visit_function_like(node, "async_function")

    return results


def analyze_docstrings(source: str, path: str | None = None) -> list[DocstringInfo]:
    path_context = classify_docstring_path_context(path)
    usage_signals = find_docstring_usage_signals(source)
    infos = iter_docstrings(source)

    for info in infos:
        remove, reasons = decide_docstring_removal(
            info,
            path_context=path_context,
            usage_signals=usage_signals,
        )
        info.remove_candidate = remove
        info.risk_reasons = reasons

    return infos


def _line_start_offsets(text: str) -> list[int]:
    offs = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offs.append(i + 1)
    return offs


def _ast_span_to_slice(text: str, lineno: int, col: int, end_lineno: int, end_col: int) -> tuple[int, int]:
    offs = _line_start_offsets(text)
    if lineno < 1 or lineno > len(offs):
        raise ValueError("bad lineno")
    start = offs[lineno - 1] + col
    end_line_idx = end_lineno - 1
    if end_line_idx >= len(offs):
        end = len(text)
    else:
        end = offs[end_line_idx] + end_col
    return start, end


def _path_context_report(path: str | None) -> dict[str, Any]:
    ctx = classify_docstring_path_context(path)
    return {
        "normalized_path": ctx["normalized_path"],
        "matched_context_labels": list(ctx["matched_context_labels"]),
        "is_low_risk_context": ctx["is_low_risk_context"],
    }


def remove_safe_docstrings(
    source: str,
    path: str | None = None,
    *,
    retain_if: Optional[Callable[["DocstringInfo"], bool]] = None,
) -> tuple[str, dict[str, Any]]:
    path_context = classify_docstring_path_context(path)
    path_report = _path_context_report(path)

    try:
        infos = analyze_docstrings(source, path=path)
    except SyntaxError:
        return source, {
            "removed_count": 0,
            "kept_count": 0,
            "removed": [],
            "kept": [],
            "parse_failed": True,
            "path_context": path_report,
            "b_probe_retained_count": 0,
            "b_probe_retained_chars": 0,
        }

    try:
        from stage3.lexical.string_classifier import SemanticClassifierConfig, classify_semantic_free_text

        _b_cfg = SemanticClassifierConfig()
    except Exception:
        classify_semantic_free_text = None  # type: ignore[assignment]
        _b_cfg = None

    probe_retained_chars = 0
    probe_retained_count = 0
    if retain_if is not None:
        for info in infos:
            if info.remove_candidate and retain_if(info):
                probe_retained_count += 1
                probe_retained_chars += len(info.text)

    to_remove = [
        i
        for i in infos
        if i.remove_candidate and not (retain_if is not None and retain_if(i))
    ]

    spans: list[tuple[int, int, DocstringInfo]] = []
    for info in to_remove:
        try:
            a, b = _ast_span_to_slice(
                source,
                info.lineno,
                info.col_offset,
                info.end_lineno,
                info.end_col_offset,
            )
            spans.append((a, b, info))
        except Exception:
            info.remove_candidate = False
            info.risk_reasons.append("span_mapping_failed_preserve")

    spans.sort(key=lambda t: t[0], reverse=True)
    new_src = source
    removed_report: list[dict[str, Any]] = []
    for a, b, info in spans:
        if a < 0 or b > len(new_src) or a >= b:
            continue
        inner_chars = len(info.text)
        routes_b = False
        if classify_semantic_free_text is not None and _b_cfg is not None:
            try:
                routes_b = classify_semantic_free_text(repr(info.text), _b_cfg)[0] == "B"
            except Exception:
                routes_b = False
        removed_report.append(
            {
                "qualname": info.qualname,
                "kind": info.kind,
                "lineno": info.lineno,
                "end_lineno": info.end_lineno,
                "risk_reasons": list(info.risk_reasons),
                "inner_chars": inner_chars,
                "routes_b_candidate": routes_b,
            }
        )
        new_src = new_src[:a] + new_src[b:]

    kept_report = [
        {
            "qualname": i.qualname,
            "kind": i.kind,
            "lineno": i.lineno,
            "end_lineno": i.end_lineno,
            "risk_reasons": i.risk_reasons,
        }
        for i in infos
        if not i.remove_candidate
    ]

    return new_src, {
        "removed_count": len(removed_report),
        "kept_count": len(kept_report),
        "removed": removed_report,
        "kept": kept_report,
        "parse_failed": False,
        "path_context": path_report,
        "b_probe_retained_count": probe_retained_count,
        "b_probe_retained_chars": probe_retained_chars,
    }
