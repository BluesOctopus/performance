#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimPy reproducible pipeline

This script consolidates a Colab-style workflow into a single, repeatable Python program:

1) Prepare SimPy repo directory (optionally unzip SimPy-master.zip).
2) Build Tree-sitter language library required by SimPy (spy/build/languages.so).
3) (Optional) Run a small Transformer parse/decode sanity check.
4) Create a small, disk-friendly sample dataset from `bigcode/starcoderdata` using streaming,
   filtered by stars >= threshold, and saved to `cached/starcoderdata_100star`.
5) Run token counting for a list of tokenizer backends (HF tokenizers + tiktoken codex/gpt4),
   comparing raw code tokens vs. SimPy-parsed tokens, saving results to CSV.

Notes
-----
- Access to `bigcode/starcoderdata` is gated on Hugging Face. Authentication is required.
  The script reads an access token from:
    1) HF_TOKEN / HUGGINGFACE_HUB_TOKEN environment variables, or
    2) local huggingface_hub cache via HfFolder.get_token().
- This script does not embed or print any tokens.
- Models that are gated/unavailable are skipped by default.

Typical Colab usage
-------------------
1) Upload SimPy-master.zip to /content.
2) In a notebook cell, run:
     from huggingface_hub import notebook_login
     notebook_login()
3) Run:
     !python simpy_pipeline.py --zip /content/SimPy-master.zip --all

Authoring style
---------------
- Comments and messages are technical and neutral (no conversational tone).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

# -----------------------------
# Environment / shell utilities
# -----------------------------

def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    """Run a command with inherited stdio for visibility."""
    print(f"$ {' '.join(map(str, cmd))}")
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_default_workdir() -> Path:
    """Prefer /content when present (Colab), otherwise current directory."""
    return Path("/content") if Path("/content").exists() else Path.cwd()


def unzip_if_needed(zip_path: Path, dest_dir: Path) -> Path:
    """
    Unzip SimPy-master.zip into dest_dir when the project directory is missing.
    Returns the project root directory path.
    """
    candidate = dest_dir / "SimPy-master"
    if candidate.exists():
        return candidate

    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    # Use system unzip when available; otherwise fallback to Python zipfile.
    try:
        run_cmd(["unzip", "-q", str(zip_path)], cwd=dest_dir)
    except Exception:
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

    if candidate.exists():
        return candidate

    # Handle case-insensitive or alternate folder name.
    for p in dest_dir.iterdir():
        if p.is_dir() and p.name.lower().endswith("simpy-master"):
            return p

    raise RuntimeError(f"unzip completed but SimPy-master directory not found under: {dest_dir}")


# -----------------------------
# Hugging Face auth + caching
# -----------------------------

def set_hf_cache_under(project_dir: Path) -> None:
    """
    Place Hugging Face cache under the project directory to reduce /root cache usage.
    """
    hf_home = project_dir / "cached" / "hf"
    ensure_dir(hf_home)
    os.environ["HF_HOME"] = str(hf_home)

    # Datasets cache can be separated; transformers will also read HF_HOME in recent versions.
    datasets_cache = hf_home / "datasets"
    ensure_dir(datasets_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)

    # Keep compatibility with older transformers; safe to omit if undesired.
    transformers_cache = hf_home / "transformers"
    ensure_dir(transformers_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))


def get_hf_token() -> Optional[str]:
    """
    Return a Hugging Face token if available. Token is not printed.
    Priority:
      1) HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars
      2) huggingface_hub local cache (HfFolder.get_token)
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    try:
        from huggingface_hub import HfFolder  # type: ignore
        return HfFolder.get_token()
    except Exception:
        return None


def require_hf_token() -> str:
    """
    Require an auth token for gated resources. Raises with actionable guidance.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "Hugging Face auth token not found.\n"
            "Provide authentication via one of the following:\n"
            "- Set HF_TOKEN (preferred) or HUGGINGFACE_HUB_TOKEN environment variable.\n"
            "- In a notebook environment, run `from huggingface_hub import notebook_login; notebook_login()`.\n"
        )
    # Export to env for downstream libraries.
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


# -----------------------------
# Tree-sitter build for SimPy
# -----------------------------

def ensure_tree_sitter_built(project_dir: Path) -> Path:
    """
    Build spy/build/languages.so using tree-sitter Language.build_library.
    """
    so_path = project_dir / "spy" / "build" / "languages.so"
    if so_path.exists():
        return so_path

    ensure_dir(so_path.parent)

    # Ensure correct working directory so relative paths resolve.
    code = (
        "from tree_sitter import Language\n"
        "Language.build_library('spy/build/languages.so', ['spy_grammar'])\n"
        "print('built spy/build/languages.so')\n"
    )
    run_cmd([sys.executable, "-c", code], cwd=project_dir)

    if not so_path.exists():
        raise RuntimeError("Tree-sitter build failed: spy/build/languages.so not found.")
    return so_path


def transformer_sanity_check(project_dir: Path) -> None:
    """
    Run a minimal parse/decode round-trip test to validate the environment.
    """
    sys.path.insert(0, str(project_dir))

    try:
        from spy.parser import Transformer  # type: ignore
    except Exception:
        # Fallback import path used in some versions.
        from spy import Transformer  # type: ignore

    t = Transformer()
    code = (
        "def add(a, b):\n"
        "    if a > b:\n"
        "        return a - b\n"
        "    return a + b\n"
    )

    spy_code = t.parse(code)
    decoded = t.decode(spy_code)

    print("=== Transformer sanity check ===")
    print("original:\n", code)
    print("spy:\n", spy_code)
    print("decoded:\n", decoded)


# -----------------------------
# Streaming sampling from StarCoderData
# -----------------------------

STAR_KEYS = (
    "max_stars_count",
    "max_stars_repo_stars",
    "stars",
    "repo_stars",
)

def _extract_stars(ex: dict) -> Optional[int]:
    for k in STAR_KEYS:
        if k in ex and ex[k] is not None:
            try:
                return int(ex[k])
            except Exception:
                pass
    return None


def _load_dataset_streaming(repo_id: str, data_dir: str, split: str, token: Optional[str]):
    """
    Compatibility wrapper across datasets versions:
    - Newer: token=...
    - Older: use_auth_token=...
    """
    from datasets import load_dataset  # type: ignore

    kwargs = dict(path=repo_id, data_dir=data_dir, split=split, streaming=True)

    if token:
        # Try new signature first.
        try:
            return load_dataset(**kwargs, token=token)
        except TypeError:
            return load_dataset(**kwargs, use_auth_token=token)
    return load_dataset(**kwargs)


def create_small_starcoder_sample(
    project_dir: Path,
    lang: str,
    min_stars: int,
    max_examples: int,
    shuffle_buffer: int,
    seed: int,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """
    Create a small sample dataset under output_dir using streaming to avoid large downloads.
    """
    if output_dir.exists() and not force:
        return output_dir

    token = require_hf_token()

    from datasets import Dataset  # type: ignore
    from tqdm.auto import tqdm  # type: ignore

    ds = _load_dataset_streaming(
        repo_id="bigcode/starcoderdata",
        data_dir=lang,
        split="train",
        token=token,
    )

    if shuffle_buffer and shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    out = []
    for ex in tqdm(ds, desc=f"stream({lang}) stars>={min_stars}"):
        stars = _extract_stars(ex)
        if stars is None or stars < min_stars:
            continue

        out.append(
            {
                "content": ex.get("content", ""),
                "max_stars_count": stars,
                "id": ex.get("id"),
                "max_stars_repo_path": ex.get("max_stars_repo_path"),
                "max_stars_repo_name": ex.get("max_stars_repo_name"),
            }
        )

        if len(out) >= max_examples:
            break

    if not out:
        raise RuntimeError(
            "No examples collected.\n"
            "Possible causes:\n"
            "- No access to gated dataset `bigcode/starcoderdata` for this account/token.\n"
            "- Stars field names changed; adjust STAR_KEYS accordingly.\n"
        )

    ensure_dir(output_dir.parent)
    Dataset.from_list(out).save_to_disk(str(output_dir))
    return output_dir


# -----------------------------
# Token counting
# -----------------------------

@dataclass(frozen=True)
class TokenCountResult:
    model: str
    code_tokens: int
    filtered_code_tokens: int
    parsed_tokens: int

    @property
    def parsed_over_filtered(self) -> float:
        return (self.parsed_tokens / self.filtered_code_tokens) if self.filtered_code_tokens else float("nan")


def _is_gated_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("gated repo" in msg) or ("not in the authorized list" in msg) or ("403" in msg) or ("forbidden" in msg)


def _load_transformer(project_dir: Path):
    sys.path.insert(0, str(project_dir))
    try:
        from spy.parser import Transformer  # type: ignore
        return Transformer()
    except Exception:
        from spy import Transformer  # type: ignore
        return Transformer()


def _load_dataset_from_disk(dataset_dir: Path):
    from datasets import load_from_disk  # type: ignore
    return load_from_disk(str(dataset_dir))


def _make_tiktoken_encoding(base_name: str, spy_tokens: Sequence[str], offset: int, encoding_name: str):
    import tiktoken  # type: ignore
    base = tiktoken.encoding_for_model(base_name)
    special = {**base._special_tokens, **{tok: i + offset for i, tok in enumerate(spy_tokens)}}
    return tiktoken.Encoding(
        name=encoding_name,
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special,
    )


def _encode_len(tokenizer, text: str, is_tiktoken: bool) -> int:
    if not is_tiktoken:
        return len(tokenizer.encode(text))

    allowed = set(tokenizer._special_tokens.keys()) | {"<|endoftext|>"}
    return len(tokenizer.encode(text, allowed_special=allowed))


def token_count(
    project_dir: Path,
    dataset_dir: Path,
    models: Sequence[str],
    n_samples: int,
    output_csv: Path,
    append: bool,
    skip_on_error: bool,
) -> list[TokenCountResult]:
    """
    Replicates the original token_count.py logic with safer error handling and no hardcoded tokens.
    """
    transformer = _load_transformer(project_dir)
    ds = _load_dataset_from_disk(dataset_dir)

    # Remove first line if it looks like a special-token header line.
    re_first_line = re.compile(r"^.*\n")

    ensure_dir(output_csv.parent)
    file_exists = output_csv.exists()

    results: list[TokenCountResult] = []

    # Initialize CSV if needed.
    if (not append) or (not file_exists):
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "code_tokens", "filtered_code_tokens", "parsed_tokens", "parsed_over_filtered"])

    from tqdm.auto import tqdm  # type: ignore
    import transformers  # type: ignore

    for model in models:
        print(f"\n=== model: {model} ===")

        is_tk = model in {"codex", "gpt4"}

        try:
            if model == "gpt4":
                tokenizer = _make_tiktoken_encoding(
                    base_name="gpt-4",
                    spy_tokens=transformer.special_tokens,
                    offset=100264,
                    encoding_name="gpt4-spy",
                )
            elif model == "codex":
                tokenizer = _make_tiktoken_encoding(
                    base_name="code-davinci-002",
                    spy_tokens=transformer.special_tokens,
                    offset=50281,
                    encoding_name="codex-spy",
                )
            else:
                # HF model/tokenizer load may fail due to gating or missing trust_remote_code.
                tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                tokenizer.add_special_tokens({"additional_special_tokens": transformer.special_tokens})
        except Exception as e:
            if skip_on_error:
                tag = "GATED" if _is_gated_error(e) else "LOAD_FAIL"
                print(f"[{tag}] tokenizer init skipped: {e}")
                continue
            raise

        code_tokens = 0
        filtered_code_tokens = 0
        parsed_tokens = 0

        # Use slicing if dataset supports it; otherwise iterate with islice.
        contents: Iterable[str]
        try:
            contents = ds["content"][:n_samples]
        except Exception:
            contents = (ds[i]["content"] for i in range(min(n_samples, len(ds))))

        for sample in tqdm(contents, total=n_samples):
            try:
                origin_len = _encode_len(tokenizer, sample, is_tk)
            except Exception:
                # Tokenizer edge cases; count as skipped for filtered totals.
                continue

            code_tokens += origin_len

            m = re_first_line.match(sample)
            if not m:
                continue

            first_line = m.group()
            if first_line.startswith("<"):
                sample = re_first_line.sub("", sample)

            if sample.startswith("#!/"):
                continue

            try:
                spy_code = transformer.parse(sample)
            except (ValueError, RecursionError):
                continue

            try:
                parsed_len = _encode_len(tokenizer, spy_code, is_tk)
            except Exception:
                continue

            parsed_tokens += parsed_len
            filtered_code_tokens += origin_len

        res = TokenCountResult(
            model=model,
            code_tokens=code_tokens,
            filtered_code_tokens=filtered_code_tokens,
            parsed_tokens=parsed_tokens,
        )
        results.append(res)

        print(f"Code Tokens: {res.code_tokens}")
        print(f"Filtered Code Tokens: {res.filtered_code_tokens}")
        print(f"Parsed Tokens: {res.parsed_tokens}")
        print(f"Parsed / Filtered: {res.parsed_over_filtered}")

        with output_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([res.model, res.code_tokens, res.filtered_code_tokens, res.parsed_tokens, res.parsed_over_filtered])

    return results


# -----------------------------
# CLI
# -----------------------------

DEFAULT_MODELS = (
    "codex",
    "Salesforce/codegen-350M-mono",
    "bigcode/santacoder",
    "Salesforce/codegen2-7B",
    # Additional models can be added; gated/unavailable ones are skipped by default.
    # "bigcode/starcoder",
    # "gpt4",
    # "gpt2",
)

def parse_models_arg(models_arg: Optional[str]) -> list[str]:
    if not models_arg:
        return list(DEFAULT_MODELS)
    # Comma-separated list
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default="", help="Optional: path to SimPy-master.zip to unzip.")
    ap.add_argument("--project-dir", type=str, default="", help="Path to SimPy-master directory.")
    ap.add_argument("--workdir", type=str, default="", help="Working directory for unzip (default: /content or cwd).")

    ap.add_argument("--install-deps", action="store_true", help="Install required Python packages via pip.")
    ap.add_argument("--build-grammar", action="store_true", help="Build tree-sitter language library.")
    ap.add_argument("--sanity-check", action="store_true", help="Run Transformer parse/decode sanity check.")

    ap.add_argument("--sample", action="store_true", help="Create a small streaming sample dataset.")
    ap.add_argument("--lang", type=str, default="python", help="Language subdir for starcoderdata (default: python).")
    ap.add_argument("--min-stars", type=int, default=100)
    ap.add_argument("--max-examples", type=int, default=5000)
    ap.add_argument("--shuffle-buffer", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-sample", action="store_true", help="Overwrite existing sampled dataset.")

    ap.add_argument("--count", action="store_true", help="Run token counting.")
    ap.add_argument("--models", type=str, default="", help="Comma-separated model list.")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--output-csv", type=str, default="", help="CSV output path (default: results/token_count.csv).")
    ap.add_argument("--append", action="store_true", help="Append rows to existing CSV instead of rewriting header.")
    ap.add_argument("--no-skip", action="store_true", help="Do not skip model/tokenizer load failures.")
    ap.add_argument("--all", action="store_true", help="Run build + sanity-check + sample + count.")

    args = ap.parse_args()

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else detect_default_workdir()

    # Determine project directory.
    project_dir: Optional[Path] = Path(args.project_dir).expanduser().resolve() if args.project_dir else None
    if not project_dir or not project_dir.exists():
        if args.zip:
            zip_path = Path(args.zip).expanduser().resolve()
            project_dir = unzip_if_needed(zip_path, workdir)
        else:
            # Best-effort defaults.
            candidates = [workdir / "SimPy-master", Path.cwd() / "SimPy-master"]
            for c in candidates:
                if c.exists():
                    project_dir = c.resolve()
                    break

    if not project_dir or not project_dir.exists():
        raise FileNotFoundError(
            "SimPy-master directory not found.\n"
            "Provide --project-dir or --zip to unzip the repository.\n"
        )

    # Cache placement reduces disk pressure in Colab.
    set_hf_cache_under(project_dir)

    if args.install_deps:
        # Minimal set used by this script.
        run_cmd(
            [sys.executable, "-m", "pip", "install", "-q",
             "tree-sitter==0.20.2",
             "datasets",
             "transformers",
             "huggingface_hub",
             "tqdm",
             "tiktoken"],
            cwd=project_dir,
        )

    do_all = args.all

    if args.build_grammar or do_all:
        ensure_tree_sitter_built(project_dir)

    if args.sanity_check or do_all:
        transformer_sanity_check(project_dir)

    dataset_dir = project_dir / "cached" / "starcoderdata_100star"
    if args.sample or do_all:
        create_small_starcoder_sample(
            project_dir=project_dir,
            lang=args.lang,
            min_stars=args.min_stars,
            max_examples=args.max_examples,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            output_dir=dataset_dir,
            force=args.force_sample,
        )
        # Size information for quick verification
        try:
            size_bytes = sum(p.stat().st_size for p in dataset_dir.rglob("*") if p.is_file())
            print(f"[OK] sample dataset written: {dataset_dir} ({size_bytes/1024/1024:.1f} MB)")
        except Exception:
            pass

    if args.count or do_all:
        models = parse_models_arg(args.models)
        out_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else (project_dir / "results" / "token_count.csv")
        token_count(
            project_dir=project_dir,
            dataset_dir=dataset_dir,
            models=models,
            n_samples=args.n_samples,
            output_csv=out_csv,
            append=args.append,
            skip_on_error=not args.no_skip,
        )
        print(f"[OK] results saved: {out_csv}")


if __name__ == "__main__":
    main()
