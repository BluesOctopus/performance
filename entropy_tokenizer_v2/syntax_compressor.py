"""Stage 1: AST skeleton → ``<SYN_N>`` + slot values; MDL greedy selection over candidates."""

import ast
import copy
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from config import (
    AST_MIN_FREQ,
    MDL_CODEBOOK_OVERHEAD,
    STAGE1_MIN_AVG_NET_SAVING,
    STAGE1_MIN_OCCURRENCES,
    STAGE1_MIN_TOTAL_NET_SAVING,
    VOCAB_COST_MODE,
)
from marker_count import encode as mc_encode
from markers import make_syn_marker
from placeholder_accounting import compute_vocab_intro_cost, count_base_tokens, count_sequence_tokens

_SKIP_NODE_TYPES = (ast.Pass, ast.Break, ast.Continue)
_TRY_TYPES = tuple(
    getattr(ast, n) for n in ("Try", "TryStar") if hasattr(ast, n)
)


def _compute_skeleton(node: ast.AST) -> Optional[str]:
    """Statement header as anonymised template ``...{0}...``; None if unsupported."""
    if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
        return None
    if isinstance(node, _SKIP_NODE_TYPES + _TRY_TYPES):
        return None

    nc   = copy.deepcopy(node)
    ctr  = [0]

    def _ph() -> str:
        idx = ctr[0]; ctr[0] += 1
        return f"_PH{idx}_"

    for attr in ("body", "orelse", "finalbody"):
        if getattr(nc, attr, None):
            setattr(nc, attr, [ast.Pass()])
    if getattr(nc, "handlers", None):
        nc.handlers = []
    if hasattr(nc, "decorator_list"):
        nc.decorator_list = []

    if isinstance(nc, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nc.name = _ph()
        for arg_list in (nc.args.args, nc.args.posonlyargs, nc.args.kwonlyargs):
            for arg in arg_list:
                arg.arg = _ph()
        if nc.args.vararg:  nc.args.vararg.arg = _ph()
        if nc.args.kwarg:   nc.args.kwarg.arg  = _ph()
    elif isinstance(nc, ast.ClassDef):
        nc.name = _ph()
    elif isinstance(nc, ast.ExceptHandler) and nc.name:
        nc.name = _ph()
    elif isinstance(nc, ast.Global):
        nc.names = [_ph() for _ in nc.names]
    elif isinstance(nc, ast.Nonlocal):
        nc.names = [_ph() for _ in nc.names]
    elif isinstance(nc, ast.ImportFrom):
        if nc.module: nc.module = _ph()
        for a in nc.names:
            a.name = _ph()
            if a.asname: a.asname = _ph()
    elif isinstance(nc, ast.Import):
        for a in nc.names:
            a.name = _ph()
            if a.asname: a.asname = _ph()

    class _Anon(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = _ph(); return n
        def visit_Constant(self, n):
            if n.value in (None, True, False): return n
            try:    n.value = _ph()
            except: pass
            return n

    try:
        nc = _Anon().visit(nc)
        ast.fix_missing_locations(nc)
        text = ast.unparse(nc)
    except Exception:
        return None

    skeleton = text.split("\n")[0]
    for i in range(ctr[0]):
        skeleton = skeleton.replace(f"'_PH{i}_'", f"{{{i}}}")
        skeleton = skeleton.replace(f'"_PH{i}_"', f"{{{i}}}")
        skeleton = skeleton.replace(f"_PH{i}_",   f"{{{i}}}")

    stripped = skeleton.strip()
    if not stripped or len(stripped) < 4:
        return None
    return skeleton


def _src_seg(source: str, node: Optional[ast.AST]) -> Optional[str]:
    """``ast.get_source_segment`` for *node*, or None."""
    if node is None:
        return None
    try:
        return ast.get_source_segment(source, node)
    except Exception:
        return None


def _extract_slots_from_source(source: str, node: ast.AST) -> list[str]:
    """Slot values from source segments, else ``ast.unparse``."""
    slots: list[str] = []
    lines = source.splitlines()

    def seg(n: Optional[ast.AST], fallback: str) -> str:
        s = _src_seg(source, n)
        return s if s is not None else fallback

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        line0 = lines[node.lineno - 1] if lines else ""
        if isinstance(node, ast.AsyncFunctionDef):
            m = re.search(r"\basync\s+def\s+([A-Za-z_]\w*)\s*\(", line0)
        else:
            m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", line0)
        slots.append(m.group(1) if m else node.name)
        for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            slots.append(seg(a, a.arg))
        if node.args.vararg:
            slots.append(seg(node.args.vararg, node.args.vararg.arg))
        if node.args.kwarg:
            slots.append(seg(node.args.kwarg, node.args.kwarg.arg))

    elif isinstance(node, ast.ClassDef):
        line0 = lines[node.lineno - 1] if lines else ""
        m = re.search(r"\bclass\s+([A-Za-z_]\w*)\b", line0)
        slots.append(m.group(1) if m else node.name)
        for base in node.bases:
            slots.append(seg(base, ast.unparse(base)))

    elif isinstance(node, (ast.For, ast.AsyncFor)):
        slots.append(seg(node.target, ast.unparse(node.target)))
        slots.append(seg(node.iter, ast.unparse(node.iter)))

    elif isinstance(node, ast.While):
        slots.append(seg(node.test, ast.unparse(node.test)))

    elif isinstance(node, ast.If):
        slots.append(seg(node.test, ast.unparse(node.test)))

    elif isinstance(node, ast.With):
        for item in node.items:
            slots.append(seg(item.context_expr, ast.unparse(item.context_expr)))
            if item.optional_vars:
                slots.append(
                    seg(item.optional_vars, ast.unparse(item.optional_vars))
                )

    elif isinstance(node, ast.Return):
        if node.value:
            slots.append(seg(node.value, ast.unparse(node.value)))

    elif isinstance(node, ast.Assign):
        for t in node.targets:
            slots.append(seg(t, ast.unparse(t)))
        slots.append(seg(node.value, ast.unparse(node.value)))

    elif isinstance(node, ast.AugAssign):
        slots.append(seg(node.target, ast.unparse(node.target)))
        slots.append(seg(node.value, ast.unparse(node.value)))

    elif isinstance(node, ast.AnnAssign):
        slots.append(seg(node.target, ast.unparse(node.target)))
        slots.append(seg(node.annotation, ast.unparse(node.annotation)))
        if node.value:
            slots.append(seg(node.value, ast.unparse(node.value)))

    elif isinstance(node, ast.Expr):
        slots.append(seg(node.value, ast.unparse(node.value)))

    elif isinstance(node, ast.Import):
        for alias in node.names:
            s = _src_seg(source, alias)
            if s:
                slots.append(s)
            elif alias.asname:
                slots.extend([alias.name, alias.asname])
            else:
                slots.append(alias.name)

    elif isinstance(node, ast.ImportFrom):
        if node.module:
            slots.append(node.module)
        for alias in node.names:
            s = _src_seg(source, alias)
            if s:
                slots.append(s)
            elif alias.asname:
                slots.extend([alias.name, alias.asname])
            else:
                slots.append(alias.name)

    elif isinstance(node, ast.Raise):
        if node.exc:
            slots.append(seg(node.exc, ast.unparse(node.exc)))

    elif isinstance(node, ast.Assert):
        slots.append(seg(node.test, ast.unparse(node.test)))
        if node.msg:
            slots.append(seg(node.msg, ast.unparse(node.msg)))

    elif isinstance(node, ast.Delete):
        for t in node.targets:
            slots.append(seg(t, ast.unparse(t)))

    elif isinstance(node, ast.ExceptHandler):
        if node.type:
            slots.append(seg(node.type, ast.unparse(node.type)))
        if node.name:
            slots.append(node.name)

    return slots


def _extract_header_source(source: str, node: ast.AST) -> str:
    """Header line(s) for *node* from *source*."""
    lines = source.splitlines()
    start = node.lineno
    end = _find_header_end_line(node, len(lines))
    return "\n".join(lines[start - 1 : end])


def estimate_baseline_token_cost(
    match_text: str,
    tokenizer,
    tok_type: str,
) -> int:
    return count_sequence_tokens(match_text, tokenizer=tokenizer, tok_type=tok_type)


def serialize_stage1_compressed_form(marker: str, slots: list[str]) -> str:
    slot_str = " ".join(slots).strip()
    return f"{marker} {slot_str}".strip()


def estimate_stage1_compressed_cost(
    marker: str,
    slots: list[str],
    tokenizer,
    tok_type: str,
) -> int:
    compressed = serialize_stage1_compressed_form(marker, slots)
    return count_base_tokens(compressed, tokenizer=tokenizer, tok_type=tok_type)


def estimate_stage1_occurrence_sequence_cost(
    match_text: str,
    marker: str,
    slots: list[str],
    *,
    tokenizer,
    tok_type: str,
) -> dict:
    baseline_sequence_tokens = count_base_tokens(
        match_text, tokenizer=tokenizer, tok_type=tok_type
    )
    compressed_text = serialize_stage1_compressed_form(marker, slots)
    compressed_sequence_tokens = count_base_tokens(
        compressed_text, tokenizer=tokenizer, tok_type=tok_type
    )
    slot_text = " ".join(slots).strip()
    slot_sequence_tokens = (
        count_base_tokens(slot_text, tokenizer=tokenizer, tok_type=tok_type)
        if slot_text
        else 0
    )
    marker_sequence_tokens = count_base_tokens(marker, tokenizer=tokenizer, tok_type=tok_type)
    return {
        "baseline_sequence_tokens": baseline_sequence_tokens,
        "compressed_sequence_tokens": compressed_sequence_tokens,
        "sequence_net_saving": baseline_sequence_tokens - compressed_sequence_tokens,
        "marker_sequence_tokens": marker_sequence_tokens,
        "slot_sequence_tokens": slot_sequence_tokens,
        "compressed_text": compressed_text,
    }


def build_stage1_vocab_entry(marker: str, skeleton: str) -> dict:
    return {"token": marker, "kind": "stage1", "definition": skeleton}


def estimate_stage1_candidate_effective_gain(
    candidate_skeleton: str,
    occurrences: list[tuple[str, list[str]]],
    marker: str,
    *,
    tokenizer,
    tok_type: str,
    vocab_cost_mode: str = VOCAB_COST_MODE,
) -> dict:
    if not occurrences:
        entry = build_stage1_vocab_entry(marker, candidate_skeleton)
        vocab_intro = compute_vocab_intro_cost(
            [entry], mode=vocab_cost_mode, tokenizer=tokenizer, tok_type=tok_type
        )
        return {
            "marker": marker,
            "skeleton": candidate_skeleton,
            "candidate_occurrences": 0,
            "total_baseline_sequence_tokens": 0,
            "total_compressed_sequence_tokens": 0,
            "total_sequence_net_saving": 0,
            "vocab_intro_tokens": vocab_intro,
            "effective_total_net_saving": -vocab_intro,
            "avg_sequence_net_saving": 0.0,
        }

    total_baseline = 0
    total_compressed = 0
    total_seq_net = 0
    for match_text, slots in occurrences:
        oc = estimate_stage1_occurrence_sequence_cost(
            match_text, marker, slots, tokenizer=tokenizer, tok_type=tok_type
        )
        total_baseline += int(oc["baseline_sequence_tokens"])
        total_compressed += int(oc["compressed_sequence_tokens"])
        total_seq_net += int(oc["sequence_net_saving"])

    entry = build_stage1_vocab_entry(marker, candidate_skeleton)
    vocab_intro = compute_vocab_intro_cost(
        [entry], mode=vocab_cost_mode, tokenizer=tokenizer, tok_type=tok_type
    )
    n = len(occurrences)
    avg_seq = total_seq_net / n
    effective = total_seq_net - vocab_intro
    return {
        "marker": marker,
        "skeleton": candidate_skeleton,
        "candidate_occurrences": n,
        "total_baseline_sequence_tokens": total_baseline,
        "total_compressed_sequence_tokens": total_compressed,
        "total_sequence_net_saving": total_seq_net,
        "vocab_intro_tokens": vocab_intro,
        "effective_total_net_saving": effective,
        "avg_sequence_net_saving": avg_seq,
    }


def estimate_stage1_net_saving(
    match_text: str,
    marker: str,
    slots: list[str],
    tokenizer,
    tok_type: str,
) -> int:
    baseline_cost = estimate_baseline_token_cost(match_text, tokenizer, tok_type)
    compressed_cost = estimate_stage1_compressed_cost(marker, slots, tokenizer, tok_type)
    return baseline_cost - compressed_cost


def score_skeleton_occurrence(
    match_text: str,
    skeleton: str,
    slots: list[str],
    marker: str,
    tokenizer,
    tok_type: str,
) -> dict:
    del skeleton
    oc = estimate_stage1_occurrence_sequence_cost(
        match_text, marker, slots, tokenizer=tokenizer, tok_type=tok_type
    )
    baseline_cost = int(oc["baseline_sequence_tokens"])
    compressed_cost = int(oc["compressed_sequence_tokens"])
    return {
        "baseline_cost": baseline_cost,
        "compressed_cost": compressed_cost,
        "net_saving": int(oc["sequence_net_saving"]),
        "slot_count": len(slots),
        "slot_cost": int(oc["slot_sequence_tokens"]),
        "marker_cost": int(oc["marker_sequence_tokens"]),
    }


def score_skeleton_candidate(
    skeleton: str,
    occurrences: list[tuple[str, list[str]]],
    marker: str,
    tokenizer,
    tok_type: str,
) -> dict:
    if not occurrences:
        eg = estimate_stage1_candidate_effective_gain(
            skeleton,
            [],
            marker,
            tokenizer=tokenizer,
            tok_type=tok_type,
        )
        return {
            "skeleton": skeleton,
            "occurrences": 0,
            "avg_baseline_cost": 0.0,
            "avg_compressed_cost": 0.0,
            "avg_slot_cost": 0.0,
            "marker_cost": count_base_tokens(marker, tokenizer=tokenizer, tok_type=tok_type),
            "avg_net_saving": 0.0,
            "total_net_saving": 0,
            "selected": False,
            "vocab_intro_tokens": int(eg["vocab_intro_tokens"]),
            "effective_total_net_saving": int(eg["effective_total_net_saving"]),
            "avg_sequence_net_saving": float(eg["avg_sequence_net_saving"]),
            "total_baseline_sequence_tokens": int(eg["total_baseline_sequence_tokens"]),
            "total_compressed_sequence_tokens": int(eg["total_compressed_sequence_tokens"]),
        }

    stats = [
        score_skeleton_occurrence(
            match_text,
            skeleton,
            slots,
            marker,
            tokenizer,
            tok_type,
        )
        for match_text, slots in occurrences
    ]
    n = len(stats)
    sum_baseline = sum(s["baseline_cost"] for s in stats)
    sum_compressed = sum(s["compressed_cost"] for s in stats)
    sum_slot_cost = sum(s["slot_cost"] for s in stats)
    total_net = sum(s["net_saving"] for s in stats)
    eg = estimate_stage1_candidate_effective_gain(
        skeleton,
        occurrences,
        marker,
        tokenizer=tokenizer,
        tok_type=tok_type,
    )
    return {
        "skeleton": skeleton,
        "occurrences": n,
        "avg_baseline_cost": sum_baseline / n,
        "avg_compressed_cost": sum_compressed / n,
        "avg_slot_cost": sum_slot_cost / n,
        "marker_cost": count_base_tokens(marker, tokenizer=tokenizer, tok_type=tok_type),
        "avg_net_saving": total_net / n,
        "total_net_saving": total_net,
        "selected": False,
        "vocab_intro_tokens": int(eg["vocab_intro_tokens"]),
        "effective_total_net_saving": int(eg["effective_total_net_saving"]),
        "avg_sequence_net_saving": float(eg["avg_sequence_net_saving"]),
        "total_baseline_sequence_tokens": int(eg["total_baseline_sequence_tokens"]),
        "total_compressed_sequence_tokens": int(eg["total_compressed_sequence_tokens"]),
    }


def mine_skeletons(sources: list[str], min_freq: int = AST_MIN_FREQ) -> Counter:
    """Per-skeleton counts across *sources*, keeping keys with count ≥ *min_freq*."""
    counter: Counter = Counter()
    for src in sources:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            sk = _compute_skeleton(node)
            if sk:
                counter[sk] += 1
    return Counter({k: v for k, v in counter.items() if v >= min_freq})


def empirical_skeleton_token_savings(
    sources: list[str],
    skeleton_keys: set[str],
    tokenizer,
    tok_type: str,
    *,
    probe_op: str = "<SYN_0>",
) -> dict[str, tuple[int, int]]:
    """Per skeleton: (apply count, total token savings) using ``count_augmented``."""
    out: dict[str, list[int]] = {sk: [0, 0] for sk in skeleton_keys}
    occurrences = collect_skeleton_occurrences(sources, skeleton_keys)
    for sk, occ in occurrences.items():
        for before, slots in occ:
            saved = estimate_stage1_net_saving(
                before,
                probe_op,
                slots,
                tokenizer,
                tok_type,
            )
            out[sk][0] += 1
            out[sk][1] += int(saved)
    return {sk: (vals[0], vals[1]) for sk, vals in out.items()}


def _header_token_count(skeleton: str, tokenizer, tok_type: str) -> int:
    """Tokenizer length of skeleton with ``{N}`` slots stripped (proxy for fixed part)."""
    fixed = re.sub(r'\{\d+\}', '', skeleton)
    fixed = fixed.strip()
    if not fixed:
        return 1
    try:
        if tok_type == "tiktoken":
            return len(tokenizer.encode(fixed, allowed_special="all"))
        return len(tokenizer.encode(fixed, add_special_tokens=False))
    except Exception:
        return max(1, len(fixed.split()))


def _num_slots(skeleton: str) -> int:
    return len(re.findall(r'\{\d+\}', skeleton))


def collect_skeleton_occurrences(
    sources: list[str],
    skeleton_keys: set[str],
) -> dict[str, list[tuple[str, list[str]]]]:
    out: dict[str, list[tuple[str, list[str]]]] = {sk: [] for sk in skeleton_keys}
    for src in sources:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        matches: list[tuple[int, int, str, ast.AST]] = []
        for node in ast.walk(tree):
            sk = _compute_skeleton(node)
            if not sk or sk not in skeleton_keys:
                continue
            sl = node.lineno
            el = _find_header_end_line(node, len(lines))
            matches.append((sl, el, sk, node))
        matches.sort(key=lambda t: (t[0], t[1], t[2]))
        used_start: set[int] = set()
        for sl, _el, sk, node in matches:
            if sl in used_start:
                continue
            used_start.add(sl)
            before = _extract_header_source(src, node)
            slots = _extract_slots_from_source(src, node)
            out[sk].append((before, slots))
    return out


@dataclass
class SkeletonCandidate:
    skeleton:    str
    frequency:   int
    fixed_tokens: int
    num_slots:   int
    savings_per_instance: float
    codebook_cost: int
    mdl_net_benefit: float
    empirical_total_savings: int
    avg_baseline_cost: float = 0.0
    avg_compressed_cost: float = 0.0
    avg_slot_cost: float = 0.0
    marker_cost: int = 1
    marker_text: str = ""
    avg_net_saving: float = 0.0
    total_net_saving: int = 0
    selected: bool = False
    # Effective-total accounting (sequence net − vocab intro for one SYN entry).
    vocab_intro_tokens: int = 0
    effective_total_net_saving: int = 0
    total_baseline_sequence_tokens: int = 0
    total_compressed_sequence_tokens: int = 0
    avg_sequence_net_saving: float = 0.0


def build_candidate_pool(
    skeleton_counts: Counter,
    tokenizer,
    tok_type: str,
    sources: list[str],
    *,
    marker_tokens: list[str] | None = None,
    min_occurrences: int | None = None,
    min_total_net_saving: int | None = None,
    min_avg_net_saving: float | None = None,
) -> list[SkeletonCandidate]:
    """Build Stage1 candidates using net-saving under augmented counting."""
    occ_min = STAGE1_MIN_OCCURRENCES if min_occurrences is None else int(min_occurrences)
    tot_min = STAGE1_MIN_TOTAL_NET_SAVING if min_total_net_saving is None else int(min_total_net_saving)
    avg_min = float(STAGE1_MIN_AVG_NET_SAVING if min_avg_net_saving is None else min_avg_net_saving)
    keys = set(skeleton_counts.keys())
    occurrences_map = collect_skeleton_occurrences(sources, keys)
    candidates: list[SkeletonCandidate] = []
    ordered_skeletons = list(skeleton_counts.keys())
    resolved_markers = list(marker_tokens or [make_syn_marker(index) for index in range(len(ordered_skeletons))])

    for index, sk in enumerate(ordered_skeletons):
        fixed_toks = _header_token_count(sk, tokenizer, tok_type)
        cb = MDL_CODEBOOK_OVERHEAD
        occurrences = occurrences_map.get(sk, [])
        probe_marker = resolved_markers[index] if index < len(resolved_markers) else make_syn_marker(index)
        stats = score_skeleton_candidate(
            sk,
            occurrences,
            probe_marker,
            tokenizer,
            tok_type,
        )
        applied = int(stats["occurrences"])
        total_seq_saved = int(stats["total_net_saving"])
        effective_saved = int(stats.get("effective_total_net_saving", total_seq_saved))
        avg_seq_saved = float(stats.get("avg_sequence_net_saving", stats["avg_net_saving"]))
        if applied < occ_min:
            continue
        if effective_saved < tot_min:
            continue
        if avg_seq_saved < avg_min:
            continue
        spi = avg_seq_saved
        candidates.append(
            SkeletonCandidate(
                skeleton=sk,
                frequency=applied,
                fixed_tokens=fixed_toks,
                num_slots=_num_slots(sk),
                savings_per_instance=spi,
                codebook_cost=cb,
                mdl_net_benefit=float(effective_saved) - cb,
                empirical_total_savings=total_seq_saved,
                avg_baseline_cost=float(stats["avg_baseline_cost"]),
                avg_compressed_cost=float(stats["avg_compressed_cost"]),
                avg_slot_cost=float(stats["avg_slot_cost"]),
                marker_cost=int(stats["marker_cost"]),
                marker_text=probe_marker,
                avg_net_saving=float(stats["avg_net_saving"]),
                total_net_saving=total_seq_saved,
                selected=False,
                vocab_intro_tokens=int(stats.get("vocab_intro_tokens", 0)),
                effective_total_net_saving=effective_saved,
                total_baseline_sequence_tokens=int(
                    stats.get("total_baseline_sequence_tokens", 0)
                ),
                total_compressed_sequence_tokens=int(
                    stats.get("total_compressed_sequence_tokens", 0)
                ),
                avg_sequence_net_saving=avg_seq_saved,
            )
        )

    candidates.sort(
        key=lambda c: (
            c.effective_total_net_saving,
            c.total_net_saving,
            c.avg_sequence_net_saving,
            c.frequency,
        ),
        reverse=True,
    )
    return candidates


def greedy_mdl_select(
    candidates: list[SkeletonCandidate],
    N_baseline: int,
    V0: int,
    *,
    return_diagnostics: bool = False,
) -> list[SkeletonCandidate] | tuple[list[SkeletonCandidate], list[dict], float]:
    """Greedy accept by positive net saving; keep candidate order semantics."""
    N_current = N_baseline
    accepted: list[SkeletonCandidate] = []
    diagnostics: list[dict] = []

    for cand in candidates:
        if N_current <= 0:
            break
        if cand.frequency == 0:
            continue

        net_gain = int(cand.effective_total_net_saving)

        if net_gain <= 0:
            diagnostics.append({
                "skeleton": cand.skeleton,
                "accepted": False,
                "mdl_delta": 0.0,
                "total_net_saving": net_gain,
                "effective_total_net_saving": int(cand.effective_total_net_saving),
            })
            continue

        cand.selected = True
        accepted.append(cand)
        N_current = max(0, N_current - net_gain)
        diagnostics.append({
            "skeleton": cand.skeleton,
            "accepted": True,
            "mdl_delta": -float(net_gain),
            "total_net_saving": net_gain,
            "effective_total_net_saving": int(cand.effective_total_net_saving),
        })

    if return_diagnostics:
        return accepted, diagnostics, float(N_baseline - N_current)
    return accepted


def _find_header_end_line(node: ast.AST, total_lines: int) -> int:
    """Last line of statement header (before body); simple stmts → ``lineno``."""
    body_first_line: Optional[int] = None
    for attr in ("body", "handlers", "orelse"):
        children = getattr(node, attr, None)
        if children:
            child_line = getattr(children[0], "lineno", None)
            if child_line is not None:
                if body_first_line is None or child_line < body_first_line:
                    body_first_line = child_line

    if body_first_line is not None and body_first_line > node.lineno:
        return body_first_line - 1
    return node.lineno


def _collect_syntax_replacements(
    source: str,
    selected: list[SkeletonCandidate],
    tokenizer=None,
    tok_type: Optional[str] = None,
    *,
    prune_nonpositive: bool = False,
) -> tuple[dict[int, tuple[int, str]], dict[str, dict]] | None:
    """start_line → (end_line, compressed line) with occurrence-level stats; None on parse error."""
    if not selected:
        return {}, {}

    skeleton_to_op: dict[str, str] = {
        cand.skeleton: (cand.marker_text or make_syn_marker(i))
        for i, cand in enumerate(selected)
    }
    stats_by_skeleton: dict[str, dict] = {
        cand.skeleton: {
            "candidate_occurrences": 0,
            "replaced_occurrences": 0,
            "skipped_nonpositive_occurrences": 0,
            "total_net_saving_replaced": 0,
        }
        for cand in selected
    }

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines()
    replacements: dict[int, tuple[int, str]] = {}

    for node in ast.walk(tree):
        sk = _compute_skeleton(node)
        if not sk or sk not in skeleton_to_op:
            continue
        op_tok = skeleton_to_op[sk]
        slots = _extract_slots_from_source(source, node)
        compressed = serialize_stage1_compressed_form(op_tok, slots)
        stats_by_skeleton[sk]["candidate_occurrences"] += 1

        start_line = node.lineno
        end_line = _find_header_end_line(node, len(lines))

        if (
            prune_nonpositive
            and tokenizer is not None
            and tok_type is not None
        ):
            before = _extract_header_source(source, node)
            oc = estimate_stage1_occurrence_sequence_cost(
                before,
                op_tok,
                slots,
                tokenizer=tokenizer,
                tok_type=tok_type,
            )
            net_saving = int(oc["sequence_net_saving"])
            if net_saving <= 0:
                stats_by_skeleton[sk]["skipped_nonpositive_occurrences"] += 1
                continue
            stats_by_skeleton[sk]["total_net_saving_replaced"] += int(net_saving)

        if start_line not in replacements:
            replacements[start_line] = (end_line, compressed)
            stats_by_skeleton[sk]["replaced_occurrences"] += 1

    return replacements, stats_by_skeleton


def sum_replaced_header_tokens(
    source: str,
    selected: list[SkeletonCandidate],
    tokenizer,
    tok_type: str,
) -> tuple[int, int]:
    """Total tokens in replaced header spans and number of sites."""
    payload = _collect_syntax_replacements(source, selected)
    if not payload:
        return 0, 0
    repl, _stats = payload
    if not repl:
        return 0, 0
    lines = source.splitlines()
    total = 0
    for start_line, (end_line, _) in repl.items():
        header = "\n".join(lines[start_line - 1 : end_line])
        total += len(mc_encode(tokenizer, tok_type, header))
    return total, len(repl)


def compress_source_syntax(
    source: str,
    selected: list[SkeletonCandidate],
    tokenizer=None,
    tok_type: Optional[str] = None,
    *,
    prune_nonpositive: bool = False,
    return_stats: bool = False,
) -> str | tuple[str, dict[str, dict]]:
    """Replace matched statement headers with ``<SYN_N> slot ...``; bodies unchanged."""
    payload = _collect_syntax_replacements(
        source,
        selected,
        tokenizer,
        tok_type,
        prune_nonpositive=prune_nonpositive,
    )
    if payload is None:
        if return_stats:
            return source, {}
        return source
    replacements, stats_by_skeleton = payload
    if not replacements:
        if return_stats:
            return source, stats_by_skeleton
        return source

    lines = source.splitlines()

    skip_until = -1
    result: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        if lineno <= skip_until:
            continue
        if lineno in replacements:
            end_line, compressed = replacements[lineno]
            result.append(compressed)
            skip_until = end_line
        else:
            result.append(line)

    out = "\n".join(result)
    if return_stats:
        return out, stats_by_skeleton
    return out
