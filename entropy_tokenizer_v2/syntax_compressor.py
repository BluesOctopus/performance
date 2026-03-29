"""Stage 1: Single-line Header Mining & Compression.
Mines sequences of statements (Block Patterns) to achieve higher compression.
Replaces high-frequency headers with <SYN_N> + slot values.
Note: Multi-line block mining is currently disabled to ensure 100% reversibility.
"""

import ast
import copy
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Tuple

from config import AST_MIN_FREQ, MDL_CODEBOOK_OVERHEAD
from marker_count import RE_ALL_MARKERS, count_augmented, encode as mc_encode

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

def _get_stmt_skeleton(node: ast.AST) -> Optional[str]:
    """Anonymize a single statement into a skeleton string (Balanced version)."""
    if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
        return None
    
    try:
        nc = copy.deepcopy(node)
        ctr = [0]
        def _ph():
            idx = ctr[0]; ctr[0] += 1
            return f"_PH{idx}_"

        # Anonymize fields that are strings (names of functions, classes, etc.)
        if isinstance(nc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nc.name = _ph()
        if isinstance(nc, ast.ImportFrom):
            if nc.module: nc.module = _ph()
        if isinstance(nc, (ast.Import, ast.ImportFrom)):
            for alias in nc.names:
                alias.name = _ph()
                if alias.asname: alias.asname = _ph()
        if isinstance(nc, ast.ExceptHandler) and nc.name:
            nc.name = _ph()

        class _Anon(ast.NodeTransformer):
            def visit_Name(self, n):
                n.id = _ph(); return n
            def visit_Constant(self, n):
                if n.value in (None, True, False): return n
                try: n.value = _ph()
                except: pass
                return n
            def visit_arg(self, n):
                n.arg = _ph(); return n

        nc = _Anon().visit(nc)
        # Strip bodies to focus on the 'header' or 'structure' of the statement
        for attr in ("body", "orelse", "finalbody", "handlers"):
            if hasattr(nc, attr):
                setattr(nc, attr, [])
        
        text = ast.unparse(nc).strip()
        # Replace placeholders with {0}, {1}...
        for i in range(ctr[0]):
            text = text.replace(f"'_PH{i}_'", f"{{{i}}}")
            text = text.replace(f'"_PH{i}_"', f"{{{i}}}")
            text = text.replace(f"_PH{i}_", f"{{{i}}}")
        return text
    except:
        return None

def mine_block_patterns(sources: List[str], min_freq: int = AST_MIN_FREQ) -> Counter:
    """Mine single-line headers only (Multi-line blocks disabled for reversibility)."""
    counter: Counter = Counter()
    
    for src in sources:
        try:
            tree = ast.parse(src)
        except:
            continue
            
        for node in ast.walk(tree):
            if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
                continue
                
            sk = _get_stmt_skeleton(node)
            if sk:
                counter[sk] += 1
                            
    return Counter({k: v for k, v in counter.items() if v >= min_freq})

def _extract_slots_from_block(source: str, nodes: List[ast.AST]) -> List[str]:
    """Extract variable/constant values from a sequence of nodes."""
    slots = []
    
    def _add_slot(val):
        if val is not None:
            # Encode spaces to avoid splitting issues during decompression
            slots.append(str(val).replace(" ", "__SP__"))

    for node in nodes:
        # Manually extract fields that were anonymized in _get_stmt_skeleton
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _add_slot(node.name)
        if isinstance(node, ast.ImportFrom):
            if node.module: _add_slot(node.module)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                _add_slot(alias.name)
                if alias.asname: _add_slot(alias.asname)
        if isinstance(node, ast.ExceptHandler) and node.name:
            _add_slot(node.name)

        class SlotExtractor(ast.NodeVisitor):
            def visit_Name(self, n):
                _add_slot(n.id)
            def visit_Constant(self, n):
                if n.value not in (None, True, False):
                    _add_slot(n.value)
            def visit_arg(self, n):
                _add_slot(n.arg)
        
        SlotExtractor().visit(node)
    return slots

def empirical_block_savings(
    sources: List[str],
    block_keys: set,
    tokenizer,
    tok_type: str,
    probe_op: str = "<SYN_0>"
) -> dict:
    """Calculate empirical savings for single-line headers."""
    out = {sk: [0, 0] for sk in block_keys}
    
    for src in sources:
        try:
            tree = ast.parse(src)
        except: continue
        
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
                continue
                
            sk_single = _get_stmt_skeleton(node)
            if sk_single in block_keys:
                start_line = node.lineno
                before_text = lines[start_line-1] if start_line <= len(lines) else ""
                slots = _extract_slots_from_block(src, [node])
                after_text = f"{probe_op} {' '.join(slots)}"
                
                saved = count_augmented(before_text, tokenizer, tok_type) - count_augmented(after_text, tokenizer, tok_type)
                out[sk_single][0] += 1
                out[sk_single][1] += saved
                            
    return {k: (v[0], v[1]) for k, v in out.items()}

def build_candidate_pool(
    block_counts: Counter,
    tokenizer,
    tok_type: str,
    sources: List[str]
) -> List[SkeletonCandidate]:
    keys = set(block_counts.keys())
    savings_map = empirical_block_savings(sources, keys, tokenizer, tok_type)
    
    candidates = []
    for sk, freq in block_counts.items():
        applied, total_saved = savings_map.get(sk, (0, 0))
        if applied <= 0 or total_saved <= 0: continue
        
        # Estimate fixed tokens in the skeleton (excluding slots)
        fixed_text = re.sub(r'\{\d+\}', '', sk).replace(" ; ", " ")
        fixed_toks = len(mc_encode(tokenizer, tok_type, fixed_text))
        
        num_slots = len(re.findall(r'\{\d+\}', sk))
        spi = total_saved / applied
        cb = MDL_CODEBOOK_OVERHEAD
        
        candidates.append(SkeletonCandidate(
            skeleton=sk,
            frequency=applied,
            fixed_tokens=fixed_toks,
            num_slots=num_slots,
            savings_per_instance=spi,
            codebook_cost=cb,
            mdl_net_benefit=float(total_saved) - cb,
            empirical_total_savings=total_saved
        ))
    
    candidates.sort(key=lambda c: c.mdl_net_benefit, reverse=True)
    return candidates

def greedy_mdl_select(candidates: List[SkeletonCandidate], N_baseline: int, V0: int) -> List[SkeletonCandidate]:
    log2_V0 = math.log2(V0) if V0 > 1 else 1.0
    N_current = N_baseline
    S_size = 0
    accepted = []
    
    for cand in candidates:
        if N_current <= 0: break
        log2_V_curr = math.log2(V0 + S_size) if (V0 + S_size) > 1 else 1.0
        log2_V_new = math.log2(V0 + S_size + 1)
        
        N_new = N_current - cand.empirical_total_savings
        delta = N_new * log2_V_new - N_current * log2_V_curr + log2_V0
        
        if delta >= 0:
            continue
            
        accepted.append(cand)
        N_current = N_new
        S_size += 1
    return accepted

def compress_source_syntax(source: str, selected: List[SkeletonCandidate]) -> str:
    """Apply single-line header replacements to source."""
    if not selected: return source
    
    sk_to_op = {c.skeleton: (f"<SYN_{i}>", c.num_slots) for i, c in enumerate(selected)}
    try:
        tree = ast.parse(source)
    except: return source
    
    lines = source.splitlines()
    replacements = {} # start_line -> (end_line, compressed_text)
    
    for node in ast.walk(tree):
        if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
            continue
            
        sk_single = _get_stmt_skeleton(node)
        if sk_single in sk_to_op:
            start_ln = node.lineno
            # For headers, we only replace the header line(s), not the whole body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.While, ast.If, ast.With, ast.Try)):
                end_ln = start_ln 
            else:
                end_ln = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                
            if not any(ln in replacements for ln in range(start_ln, end_ln + 1)):
                op, _ = sk_to_op[sk_single]
                slots = _extract_slots_from_block(source, [node])
                replacements[start_ln] = (end_ln, f"{op} {' '.join(slots)}")
                            
    if not replacements: return source
    
    result = []
    skip_until = -1
    for i, line in enumerate(lines, start=1):
        if i <= skip_until: continue
        if i in replacements:
            end_ln, compressed = replacements[i]
            result.append(compressed)
            skip_until = end_ln
        else:
            result.append(line)
    return "\n".join(result)

def decompress_source_syntax(compressed_text: str, selected: List[SkeletonCandidate]) -> str:
    """Reverse Stage 1: Replace <SYN_N> with original header and slots."""
    if not selected: return compressed_text
    
    sk_info = []
    for c in selected:
        sk = c.skeleton
        num_slots = c.num_slots
        sk_info.append((sk, num_slots))
    
    lines = compressed_text.splitlines()
    out = []
    syn_pattern = re.compile(r'<SYN_(\d+)>')
    
    for line in lines:
        match = syn_pattern.search(line)
        if not match:
            out.append(line)
            continue
            
        idx = int(match.group(1))
        if idx >= len(sk_info):
            out.append(line)
            continue
            
        sk, num_slots = sk_info[idx]
        prefix = line[:match.start()]
        suffix = line[match.end():].strip()
        
        if num_slots > 0:
            parts = suffix.split()
            slots = parts[:num_slots]
            rest = " ".join(parts[num_slots:])
            
            if len(slots) < num_slots:
                out.append(line)
                continue
                
            try:
                slots = [s.replace("__SP__", " ") for s in slots]
                restored = sk.format(*slots)
                out.append(prefix + restored + ((" " + rest) if rest else ""))
            except:
                out.append(line)
        else:
            out.append(prefix + sk + ((" " + suffix) if suffix else ""))
            
    return "\n".join(out)
