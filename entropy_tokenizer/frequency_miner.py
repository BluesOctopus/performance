"""
Step 1: Operator candidate mining.

Two mining stages:
  1. AST Pattern Mining  — extract anonymized structural skeletons from the AST.
  2. Lexical Pattern Mining — discover frequent identifier prefixes / suffixes.

Output: cache/mining_results.json
  {
    "ast_patterns":      {skeleton_str: frequency, ...},
    "lexical_prefixes":  {prefix_str: frequency, ...},
    "lexical_suffixes":  {suffix_str: frequency, ...},
  }
"""

import ast
import copy
import json
import re
from collections import Counter
from typing import Optional

from tqdm.auto import tqdm

from config import DATA_DIR, CACHE_DIR, AST_MIN_FREQ, LEXICAL_MIN_FREQ

# Node types whose skeleton is trivial (spi = 0)
_SKIP_STMTS = (ast.Pass, ast.Break, ast.Continue)
_TRY_TYPES = tuple(
    getattr(ast, n) for n in ("Try", "TryStar") if hasattr(ast, n)
)


def _load_train_dataset():
    from datasets import load_from_disk
    return load_from_disk(str(DATA_DIR / "train"))


# ═══════════════════════════════════════════════════════════════════════════════
# Generic AST skeleton generator — zero hardcoded patterns
# ═══════════════════════════════════════════════════════════════════════════════

def _anonymize_arguments(args_node, counter: list[int]):
    """Replace all argument-name strings with placeholders."""
    for arg_list in (args_node.args, args_node.posonlyargs, args_node.kwonlyargs):
        for arg in arg_list:
            arg.arg = f"_PH{counter[0]}_"; counter[0] += 1
    if args_node.vararg:
        args_node.vararg.arg = f"_PH{counter[0]}_"; counter[0] += 1
    if args_node.kwarg:
        args_node.kwarg.arg = f"_PH{counter[0]}_"; counter[0] += 1


def _anonymize_ast_node(node) -> Optional[str]:
    """
    Fully generic skeleton generator.

    For any statement-level AST node:
      1. Strip block bodies (body / orelse / finally / handlers / decorators)
         so only the statement *header* remains.
      2. Replace every Name & non-trivial Constant leaf with {N} placeholders.
      3. Also anonymise string-attribute names (func name, class name, arg
         names, import modules, except variable, …).
      4. ast.unparse → first line → placeholder substitution.

    No node types are manually enumerated; the data decides which
    skeletons are frequent enough to become operators.
    """
    if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
        return None
    if isinstance(node, _SKIP_STMTS + _TRY_TYPES):
        return None

    nc = copy.deepcopy(node)
    ctr: list[int] = [0]

    def _ph() -> str:
        idx = ctr[0]; ctr[0] += 1
        return f"_PH{idx}_"

    # ── 1. Strip block bodies to isolate the header ───────────────
    for attr in ("body", "orelse", "finalbody"):
        if getattr(nc, attr, None):
            setattr(nc, attr, [ast.Pass()])
    if getattr(nc, "handlers", None):
        setattr(nc, "handlers", [])
    if hasattr(nc, "decorator_list"):
        nc.decorator_list = []

    # ── 2. Anonymise string-based name attributes ─────────────────
    if isinstance(nc, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nc.name = _ph()
        _anonymize_arguments(nc.args, ctr)
    elif isinstance(nc, ast.ClassDef):
        nc.name = _ph()
    elif isinstance(nc, ast.ExceptHandler) and nc.name:
        nc.name = _ph()
    elif isinstance(nc, ast.Global):
        nc.names = [_ph() for _ in nc.names]
    elif isinstance(nc, ast.Nonlocal):
        nc.names = [_ph() for _ in nc.names]
    elif isinstance(nc, ast.ImportFrom):
        if nc.module:
            nc.module = _ph()
        for a in nc.names:
            a.name = _ph()
            if a.asname: a.asname = _ph()
    elif isinstance(nc, ast.Import):
        for a in nc.names:
            a.name = _ph()
            if a.asname: a.asname = _ph()

    # ── 3. Anonymise all Name / Constant expression leaves ────────
    class _Anon(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = _ph()
            return n

        def visit_Constant(self, n):
            if n.value is None or n.value is True or n.value is False:
                return n
            try:
                n.value = _ph()
            except Exception:
                pass
            return n

    try:
        nc = _Anon().visit(nc)
        ast.fix_missing_locations(nc)
        text = ast.unparse(nc)
    except Exception:
        return None

    # ── 4. First line only + placeholder substitution ─────────────
    skeleton = text.split("\n")[0]

    for i in range(ctr[0]):
        skeleton = skeleton.replace(f"'_PH{i}_'", f"{{{i}}}")
        skeleton = skeleton.replace(f'"_PH{i}_"', f"{{{i}}}")
        skeleton = skeleton.replace(f"_PH{i}_", f"{{{i}}}")

    stripped = skeleton.strip()
    if not stripped or len(stripped) < 3:
        return None

    return skeleton

def _extract_dynamic_ast_patterns(code: str) -> list[str]:
    """
    Extract dynamic structural patterns (skeletons) from Python AST.
    Returns a list of skeleton strings found in this code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    patterns = []

    for node in ast.walk(tree):
        skeleton = _anonymize_ast_node(node)
        if skeleton:
            patterns.append(skeleton)

    return patterns

def mine_ast_patterns(
    dataset,
    min_freq: int = AST_MIN_FREQ,
    max_files: Optional[int] = None,
) -> Counter:
    """
    Count AST structural patterns across the dataset.
    Returns {skeleton_str: count}.
    """
    counter = Counter()
    contents = dataset["content"]
    if max_files:
        contents = contents[:max_files]

    for code in tqdm(contents, desc="Mining Dynamic AST patterns"):
        for pat in _extract_dynamic_ast_patterns(code):
            counter[pat] += 1

    # Filter by minimum frequency to keep only high-value skeletons
    return Counter({k: v for k, v in counter.items() if v >= min_freq})

def _extract_lexical_patterns(code: str) -> tuple[list[str], list[str]]:
    """
    Extract frequent lexical patterns from code identifiers.
    Returns (prefixes, suffixes).

    Covers:
      - snake_case prefixes: get_name -> "get_"
      - Attribute access prefixes: self.name -> "self.", os.path -> "os."
      - snake_case suffixes: user_id -> "_id", data_list -> "_list"
    """
    prefixes = []
    suffixes = []

    # snake_case prefixes: get_name -> get_
    snake_words = re.findall(r'\b([a-z]{2,})_[a-zA-Z0-9_]+\b', code)
    for stem in snake_words:
        prefixes.append(stem + "_")

    # Attribute access: self.name -> self., os.path -> os.
    attr_matches = re.findall(r'\b([a-zA-Z_]\w{1,})\.([a-zA-Z_]\w*)', code)
    for obj, _ in attr_matches:
        prefixes.append(obj + ".")

    # snake_case suffixes: user_id -> _id, data_list -> _list
    suffix_words = re.findall(r'\b[a-zA-Z0-9]+_([a-z]{2,})\b', code)
    for suf in suffix_words:
        suffixes.append("_" + suf)

    return prefixes, suffixes

def mine_lexical_patterns(
    dataset,
    min_freq: int = LEXICAL_MIN_FREQ,
    max_files: Optional[int] = None,
) -> tuple[Counter, Counter]:
    """
    Count lexical prefixes and suffixes across the dataset.
    Returns (prefix_counter, suffix_counter).
    """
    prefix_counter = Counter()
    suffix_counter = Counter()
    contents = dataset["content"]
    if max_files:
        contents = contents[:max_files]

    for code in tqdm(contents, desc="Mining Lexical Patterns"):
        prefs, sufs = _extract_lexical_patterns(code)
        for p in prefs:
            prefix_counter[p] += 1
        for s in sufs:
            suffix_counter[s] += 1

    filtered_prefixes = Counter({k: v for k, v in prefix_counter.items() if v >= min_freq})
    filtered_suffixes = Counter({k: v for k, v in suffix_counter.items() if v >= min_freq})
    return filtered_prefixes, filtered_suffixes

def run_mining(max_files: Optional[int] = None) -> dict:
    """
    Run all mining stages and cache results.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "mining_results.json"

    print("[miner] Loading dataset ...")
    dataset = _load_train_dataset()

    print("[miner] === Stage 1: Dynamic AST Pattern Mining ===")
    ast_patterns = mine_ast_patterns(dataset, max_files=max_files)
    top_ast = dict(ast_patterns.most_common(2000))
    print(f"[miner] Found {len(top_ast)} AST patterns (>= min_freq)")
    print(f"[miner] Top 20: {list(top_ast.keys())[:20]}")

    print("[miner] === Stage 2: Dynamic Lexical Pattern Mining ===")
    lex_prefixes, lex_suffixes = mine_lexical_patterns(dataset, max_files=max_files)
    top_prefix = dict(lex_prefixes.most_common(200))
    top_suffix = dict(lex_suffixes.most_common(200))
    print(f"[miner] Found {len(top_prefix)} prefixes, top: {list(top_prefix.keys())[:10]}")
    print(f"[miner] Found {len(top_suffix)} suffixes, top: {list(top_suffix.keys())[:10]}")

    results = {
        "ast_patterns": top_ast,
        "lexical_prefixes": top_prefix,
        "lexical_suffixes": top_suffix,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[miner] Results cached to {cache_path}")

    return results

if __name__ == "__main__":
    run_mining()
