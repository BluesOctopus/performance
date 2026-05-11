from __future__ import annotations

import ast
import builtins
from dataclasses import asdict, dataclass
from typing import Any

from tokenizer_utils import count_tokens

_compile_source = compile


@dataclass(frozen=True)
class AlphaRenameResult:
    renamed_text: str
    renamed_count: int
    raw_tokens: int
    renamed_tokens: int
    delta_tokens: int
    ast_equivalent: bool
    skipped_reason: str


@dataclass(frozen=True)
class AlphaPassMetadata:
    alpha_applied: bool
    alpha_renamed_count: int
    alpha_raw_tokens: int
    alpha_tokens: int
    alpha_delta_tokens: int
    alpha_guardrail_triggered: bool
    alpha_rollback_reason: str
    alpha_skipped_reason: str
    alpha_ast_ok: bool
    alpha_compile_ok: bool
    alpha_public_signature_preserved: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AlphaPassResult:
    output_text: str
    metadata: AlphaPassMetadata


def alpha_rename_function_chunk(
    text: str,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
) -> AlphaRenameResult:
    raw_tokens = count_tokens(text, encoder=encoder, tok_type=tok_type)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return AlphaRenameResult(text, 0, raw_tokens, raw_tokens, 0, False, "parse_failed")

    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return AlphaRenameResult(text, 0, raw_tokens, raw_tokens, 0, False, "not_function_chunk")

    function_node = tree.body[0]
    original_signature = function_signature_dump(function_node)
    mapping = _build_rename_mapping(function_node, encoder=encoder, tok_type=tok_type)
    if not mapping:
        return AlphaRenameResult(text, 0, raw_tokens, raw_tokens, 0, True, "no_eligible_locals")

    renamed_tree = ast.fix_missing_locations(_LocalRenameTransformer(mapping).visit(tree))
    renamed_text = ast.unparse(renamed_tree)
    try:
        reparsed_tree = ast.parse(renamed_text)
    except SyntaxError:
        return AlphaRenameResult(text, 0, raw_tokens, raw_tokens, 0, False, "rename_broke_parseability")

    if not signature_preserved_between(tree, reparsed_tree):
        return AlphaRenameResult(text, 0, raw_tokens, raw_tokens, 0, False, "function_signature_changed")

    renamed_tokens = count_tokens(renamed_text, encoder=encoder, tok_type=tok_type)
    ast_equivalent = _ast_equivalent_under_mapping(tree, renamed_tree, mapping)
    if function_signature_dump(reparsed_tree.body[0]) != original_signature:
        return AlphaRenameResult(text, 0, raw_tokens, raw_tokens, 0, False, "function_signature_changed")
    return AlphaRenameResult(
        renamed_text=renamed_text,
        renamed_count=len(mapping),
        raw_tokens=raw_tokens,
        renamed_tokens=renamed_tokens,
        delta_tokens=raw_tokens - renamed_tokens,
        ast_equivalent=ast_equivalent,
        skipped_reason="",
    )


def apply_alpha_rename_pass(
    text: str,
    *,
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
) -> AlphaPassResult:
    raw_tokens = count_tokens(text, encoder=encoder, tok_type=tok_type)
    base_metadata = AlphaPassMetadata(
        alpha_applied=False,
        alpha_renamed_count=0,
        alpha_raw_tokens=raw_tokens,
        alpha_tokens=raw_tokens,
        alpha_delta_tokens=0,
        alpha_guardrail_triggered=False,
        alpha_rollback_reason="",
        alpha_skipped_reason="",
        alpha_ast_ok=False,
        alpha_compile_ok=False,
        alpha_public_signature_preserved=False,
    )
    try:
        original_tree = ast.parse(text)
    except SyntaxError:
        return AlphaPassResult(
            output_text=text,
            metadata=_replace_metadata(
                base_metadata,
                alpha_skipped_reason="parse_failed",
                alpha_rollback_reason="parse_failed",
            ),
        )

    if len(original_tree.body) != 1 or not isinstance(original_tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return AlphaPassResult(
            output_text=text,
            metadata=_replace_metadata(
                base_metadata,
                alpha_skipped_reason="not_function_chunk",
                alpha_rollback_reason="not_function_chunk",
            ),
        )

    rename_result = alpha_rename_function_chunk(
        text,
        tokenizer_name=tokenizer_name,
        encoder=encoder,
        tok_type=tok_type,
    )
    if rename_result.skipped_reason:
        return AlphaPassResult(
            output_text=text,
            metadata=_replace_metadata(
                base_metadata,
                alpha_skipped_reason=rename_result.skipped_reason,
                alpha_rollback_reason=rename_result.skipped_reason,
                alpha_ast_ok=rename_result.skipped_reason != "parse_failed",
                alpha_public_signature_preserved=rename_result.skipped_reason != "function_signature_changed",
            ),
        )

    ast_ok = False
    compile_ok = False
    signature_ok = False
    try:
        renamed_tree = ast.parse(rename_result.renamed_text)
        ast_ok = True
        signature_ok = signature_preserved_between(original_tree, renamed_tree)
    except SyntaxError:
        renamed_tree = None
        ast_ok = False
        signature_ok = False

    if ast_ok:
        try:
            _compile_source(rename_result.renamed_text, "<alpha_rename>", "exec")
            compile_ok = True
        except SyntaxError:
            compile_ok = False

    rollback_reason = ""
    guardrail = False
    output_text = rename_result.renamed_text
    if not signature_ok:
        rollback_reason = "function_signature_changed"
        guardrail = True
        output_text = text
    elif not ast_ok:
        rollback_reason = "ast_parse_failed_after_rename"
        guardrail = True
        output_text = text
    elif not compile_ok:
        rollback_reason = "compile_failed_after_rename"
        guardrail = True
        output_text = text
    elif rename_result.delta_tokens <= 0:
        rollback_reason = "no_token_gain"
        guardrail = True
        output_text = text

    applied = not guardrail
    output_tokens = rename_result.renamed_tokens if applied else raw_tokens
    return AlphaPassResult(
        output_text=output_text,
        metadata=AlphaPassMetadata(
            alpha_applied=applied,
            alpha_renamed_count=rename_result.renamed_count if applied else 0,
            alpha_raw_tokens=raw_tokens,
            alpha_tokens=output_tokens,
            alpha_delta_tokens=raw_tokens - output_tokens,
            alpha_guardrail_triggered=guardrail,
            alpha_rollback_reason=rollback_reason,
            alpha_skipped_reason="" if applied else rollback_reason,
            alpha_ast_ok=ast_ok,
            alpha_compile_ok=compile_ok,
            alpha_public_signature_preserved=signature_ok,
        ),
    )


def function_signature_dump(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    signature = {
        "name": node.name,
        "args": ast.dump(node.args, include_attributes=False),
        "returns": ast.dump(node.returns, include_attributes=False) if node.returns is not None else None,
        "decorators": [ast.dump(dec, include_attributes=False) for dec in node.decorator_list],
        "type_comment": getattr(node, "type_comment", None),
    }
    return repr(signature)


def signature_preserved_between(original_tree: ast.AST, renamed_tree: ast.AST) -> bool:
    if len(original_tree.body) != 1 or len(renamed_tree.body) != 1:
        return False
    return function_signature_dump(original_tree.body[0]) == function_signature_dump(renamed_tree.body[0])


def _replace_metadata(metadata: AlphaPassMetadata, **updates: object) -> AlphaPassMetadata:
    return AlphaPassMetadata(**{**metadata.to_dict(), **updates})


def _build_rename_mapping(function_node: ast.FunctionDef | ast.AsyncFunctionDef, *, encoder: Any, tok_type: str) -> dict[str, str]:
    collector = _LocalNameCollector()
    collector.visit(function_node)
    blocked = set(collector.arg_names) | set(collector.imported_names) | set(collector.global_names) | set(collector.nonlocal_names)
    blocked |= set(dir(builtins))
    blocked.add(function_node.name)

    eligible = [
        name
        for name in collector.local_names
        if name not in blocked and not name.startswith("__") and name != "_"
    ]
    if not eligible:
        return {}

    pool = _sorted_short_name_pool(encoder=encoder, tok_type=tok_type)
    mapping: dict[str, str] = {}
    used_targets = blocked | set(eligible)
    for name in sorted(eligible, key=lambda item: (-len(item), item)):
        for candidate in pool:
            if candidate == name or candidate in used_targets or candidate in mapping.values():
                continue
            mapping[name] = candidate
            used_targets.add(candidate)
            break
    return mapping


def _sorted_short_name_pool(*, encoder: Any, tok_type: str) -> list[str]:
    pool = list("abcdefghijk")
    return sorted(pool, key=lambda item: (count_tokens(item, encoder=encoder, tok_type=tok_type), len(item), item))


class _LocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.local_names: set[str] = set()
        self.arg_names: set[str] = set()
        self.imported_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        try:
            self._record_args(node.args)
            for child in node.body:
                self.visit(child)
        finally:
            self._depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.local_names.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.arg_names.add(node.arg)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def _record_args(self, args: ast.arguments) -> None:
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            self.arg_names.add(arg.arg)
        if args.vararg:
            self.arg_names.add(args.vararg.arg)
        if args.kwarg:
            self.arg_names.add(args.kwarg.arg)


class _LocalRenameTransformer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self._depth > 0:
            return node
        self._depth += 1
        try:
            node.body = [self.visit(child) for child in node.body]
        finally:
            self._depth -= 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node


def _ast_equivalent_under_mapping(
    original_tree: ast.AST,
    renamed_tree: ast.AST,
    mapping: dict[str, str],
) -> bool:
    inverse_mapping = {value: key for key, value in mapping.items()}
    restored_tree = ast.fix_missing_locations(_LocalRenameTransformer(inverse_mapping).visit(renamed_tree))
    return ast.dump(original_tree, include_attributes=False) == ast.dump(
        restored_tree,
        include_attributes=False,
    )
