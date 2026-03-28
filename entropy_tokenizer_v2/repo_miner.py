"""Mine per-corpus ``RepoConfig`` (Stage 1 + Stage 3 rules) and cache JSON."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from config import (
    AST_MIN_FREQ, CACHE_DIR, MDL_CODEBOOK_OVERHEAD,
    SCORE_THRESHOLD_PERCENTILE, SCORE_EPSILON,
)
from lossy_cleaner import lossless_clean
from syntax_compressor import (
    SkeletonCandidate,
    build_candidate_pool,
    greedy_mdl_select,
    mine_skeletons,
)
from marker_count import encode as _encode
from token_scorer import (
    build_replacement_map, build_vocabulary,
    compute_scores, select_replacement_set,
)


@dataclass
class RepoConfig:
    """Mined skeletons, replacement map, score summary, baseline stats."""
    selected_skeletons:  list[dict] = field(default_factory=list)
    replacement_map:     dict[str, str] = field(default_factory=dict)
    scores_summary:      list[dict] = field(default_factory=list)
    stage1_candidate_stats: list[dict] = field(default_factory=list)
    stage1_selected_stats: list[dict] = field(default_factory=list)
    stage1_total_net_saving: float = 0.0
    n_sources:           int = 0
    N_baseline_tokens:   int = 0
    V0:                  int = 0
    tokenizer_key:       str = ""

    def skeleton_candidates(self) -> list[SkeletonCandidate]:
        from dataclasses import fields

        names = [f.name for f in fields(SkeletonCandidate)]
        out: list[SkeletonCandidate] = []
        for row in self.selected_skeletons:
            d = dict(row)
            d.setdefault("vocab_intro_tokens", 0)
            d.setdefault("total_baseline_sequence_tokens", 0)
            d.setdefault("total_compressed_sequence_tokens", 0)
            d.setdefault("avg_sequence_net_saving", d.get("avg_net_saving", 0.0))
            d.setdefault(
                "effective_total_net_saving",
                d.get("total_net_saving", 0),
            )
            out.append(SkeletonCandidate(**{k: d[k] for k in names}))
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "RepoConfig":
        d = json.loads(s)
        return cls(**d)


def _load_tokenizer(tok_key: str, cfg: dict):
    if cfg["type"] == "tiktoken":
        import tiktoken
        return tiktoken.encoding_for_model(cfg["tiktoken_model"]), "tiktoken"
    import transformers
    from config import HF_TOKEN
    tok = transformers.AutoTokenizer.from_pretrained(
        cfg["name"], trust_remote_code=True, token=HF_TOKEN or None,
    )
    return tok, "hf"


def _vocab_size(tokenizer, tok_type: str) -> int:
    if tok_type == "tiktoken":
        return tokenizer.n_vocab
    vs = getattr(tokenizer, "vocab_size", None)
    if vs is not None:
        return int(vs)
    return len(tokenizer)


def collect_py_sources(repo_path: str | Path) -> list[str]:
    """All ``.py`` file contents under *repo_path*; skip unreadable."""
    sources: list[str] = []
    for root, _dirs, files in os.walk(str(repo_path)):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    sources.append(f.read())
            except OSError:
                continue
    return sources


def mine_repo(
    sources: list[str],
    tokenizer,
    tok_type: str,
    V0: int,
    tokenizer_key: str = "",
    min_freq: int = AST_MIN_FREQ,
    score_percentile: float = SCORE_THRESHOLD_PERCENTILE,
    verbose: bool = True,
) -> RepoConfig:
    n = len(sources)
    if verbose:
        print(f"[repo_miner] Mining {n} source files ...")

    # lossless_clean: blank lines + trailing ws only; keep comments/indent for parse
    clean_sources = []
    for src in sources:
        c, _ = lossless_clean(src)
        clean_sources.append(c)

    if verbose:
        print("[repo_miner] Computing baseline token counts ...")
    N_baseline = 0
    for src in tqdm(clean_sources, desc="  baseline tokens", disable=not verbose):
        N_baseline += len(_encode(tokenizer, tok_type, src))

    if verbose:
        print("[repo_miner] Stage 1 - mining AST skeletons ...")
    skeleton_counts = mine_skeletons(clean_sources, min_freq=min_freq)
    if verbose:
        print(f"  {len(skeleton_counts)} unique skeletons (freq >= {min_freq})")

    candidates = build_candidate_pool(
        skeleton_counts, tokenizer, tok_type, sources=clean_sources
    )
    selected_skeletons, _stage1_select_diag, stage1_total_net_saving = greedy_mdl_select(
        candidates,
        N_baseline,
        V0,
        return_diagnostics=True,
    )
    selected_set = {c.skeleton for c in selected_skeletons}
    stage1_candidate_stats = [
        {
            "skeleton": c.skeleton,
            "occurrences": c.frequency,
            "avg_baseline_cost": c.avg_baseline_cost,
            "avg_compressed_cost": c.avg_compressed_cost,
            "avg_slot_cost": c.avg_slot_cost,
            "marker_cost": c.marker_cost,
            "avg_net_saving": c.avg_net_saving,
            "total_net_saving": c.total_net_saving,
            "vocab_intro_tokens": c.vocab_intro_tokens,
            "effective_total_net_saving": c.effective_total_net_saving,
            "avg_sequence_net_saving": c.avg_sequence_net_saving,
            "selected": c.skeleton in selected_set,
        }
        for c in candidates
    ]
    stage1_selected_stats = [s for s in stage1_candidate_stats if s["selected"]]
    if verbose:
        print(f"  MDL K* = {len(selected_skeletons)} accepted skeletons")
        for c in selected_skeletons[:5]:
            print(
                f"    [{c.skeleton[:60]}] freq={c.frequency} "
                f"avg_net={c.avg_net_saving:.2f} total_net={c.total_net_saving}"
            )

    if verbose:
        print("[repo_miner] Stage 3 - computing token importance scores ...")
    vocab = build_vocabulary(clean_sources)
    scores = compute_scores(vocab, tokenizer, tok_type)
    replacement_set = select_replacement_set(scores, score_percentile)
    rmap = build_replacement_map(scores, replacement_set)

    if verbose:
        eligible = sum(1 for info in scores.values() if info.spt > 1.0)
        print(f"  {len(scores)} unique tokens scored, "
              f"{eligible} eligible (spt>1), "
              f"{len(replacement_set)} selected for replacement")

    from token_scorer import score_summary
    summary = score_summary(scores, top_n=50)

    return RepoConfig(
        selected_skeletons=[
            {
                "skeleton": c.skeleton,
                "frequency": c.frequency,
                "fixed_tokens": c.fixed_tokens,
                "num_slots": c.num_slots,
                "savings_per_instance": c.savings_per_instance,
                "codebook_cost": c.codebook_cost,
                "mdl_net_benefit": c.mdl_net_benefit,
                "empirical_total_savings": c.empirical_total_savings,
                "avg_baseline_cost": c.avg_baseline_cost,
                "avg_compressed_cost": c.avg_compressed_cost,
                "avg_slot_cost": c.avg_slot_cost,
                "marker_cost": c.marker_cost,
                "avg_net_saving": c.avg_net_saving,
                "total_net_saving": c.total_net_saving,
                "selected": c.selected,
                "vocab_intro_tokens": c.vocab_intro_tokens,
                "effective_total_net_saving": c.effective_total_net_saving,
                "total_baseline_sequence_tokens": c.total_baseline_sequence_tokens,
                "total_compressed_sequence_tokens": c.total_compressed_sequence_tokens,
                "avg_sequence_net_saving": c.avg_sequence_net_saving,
            }
            for c in selected_skeletons
        ],
        replacement_map=rmap,
        scores_summary=summary,
        stage1_candidate_stats=stage1_candidate_stats,
        stage1_selected_stats=stage1_selected_stats,
        stage1_total_net_saving=stage1_total_net_saving,
        n_sources=n,
        N_baseline_tokens=N_baseline,
        V0=V0,
        tokenizer_key=tokenizer_key,
    )


def mine_from_repo_path(
    repo_path: str | Path,
    tokenizer_key: str,
    tokenizer_cfg: dict,
    cache: bool = True,
    verbose: bool = True,
) -> RepoConfig:
    cache_file = CACHE_DIR / f"repo_config_{tokenizer_key}_{Path(repo_path).name}.json"
    if cache and cache_file.exists():
        if verbose:
            print(f"[repo_miner] Loading cached config: {cache_file}")
        return RepoConfig.from_json(cache_file.read_text(encoding="utf-8"))

    tokenizer, tok_type = _load_tokenizer(tokenizer_key, tokenizer_cfg)
    V0 = _vocab_size(tokenizer, tok_type)

    sources = collect_py_sources(repo_path)
    if not sources:
        raise ValueError(f"No .py files found in {repo_path}")
    if verbose:
        print(f"[repo_miner] Found {len(sources)} .py files in {repo_path}")

    config = mine_repo(sources, tokenizer, tok_type, V0,
                       tokenizer_key=tokenizer_key, verbose=verbose)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(config.to_json(), encoding="utf-8")
        if verbose:
            print(f"[repo_miner] Config saved to {cache_file}")

    return config


def mine_from_sources(
    sources: list[str],
    tokenizer_key: str,
    tokenizer_cfg: dict,
    cache_name: Optional[str] = None,
    cache: bool = True,
    verbose: bool = True,
    min_freq: int = AST_MIN_FREQ,
) -> RepoConfig:
    """Mine from a list of source strings (e.g., loaded from HF dataset)."""
    if cache and cache_name:
        cache_file = CACHE_DIR / f"repo_config_{tokenizer_key}_{cache_name}.json"
        if cache_file.exists():
            if verbose:
                print(f"[repo_miner] Loading cached config: {cache_file}")
            return RepoConfig.from_json(cache_file.read_text(encoding="utf-8"))

    tokenizer, tok_type = _load_tokenizer(tokenizer_key, tokenizer_cfg)
    V0 = _vocab_size(tokenizer, tok_type)

    config = mine_repo(sources, tokenizer, tok_type, V0,
                       tokenizer_key=tokenizer_key, verbose=verbose,
                       min_freq=min_freq)

    if cache and cache_name:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"repo_config_{tokenizer_key}_{cache_name}.json"
        cache_file.write_text(config.to_json(), encoding="utf-8")
        if verbose:
            print(f"[repo_miner] Config saved to {cache_file}")

    return config
