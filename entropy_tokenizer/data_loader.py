"""
Step 0: Stream and cache a Python code sample from StarCoderData.

Produces two splits:
  - train_split: used for frequency mining and vocabulary construction
  - test_split:  used for compression rate evaluation (never seen during mining)

Data is saved as Arrow datasets via HuggingFace datasets library.
"""

import os
import json
from pathlib import Path
from tqdm.auto import tqdm

from config import (
    HF_DATASET_REPO, HF_DATASET_LANG, HF_TOKEN,
    SAMPLE_MIN_STARS, SAMPLE_MAX_BYTES, SAMPLE_MAX_BYTES_QUICK,
    SAMPLE_SEED, SAMPLE_SHUFFLE_BUFFER, DATA_DIR,
)


STAR_KEYS = ("max_stars_count", "max_stars_repo_stars", "stars", "repo_stars")


def _extract_stars(example: dict) -> int:
    for k in STAR_KEYS:
        v = example.get(k)
        if v is not None:
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
    return 0


def stream_and_save(
    max_bytes: int | None = None,
    min_stars: int = SAMPLE_MIN_STARS,
    train_ratio: float = 0.8,
    force: bool = False,
    quick: bool = False,
):
    """
    Stream Python files from StarCoderData, collect until cumulative size
    reaches max_bytes, then split into train/test and save to disk.
    """
    if max_bytes is None:
        max_bytes = SAMPLE_MAX_BYTES_QUICK if quick else SAMPLE_MAX_BYTES

    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"

    if train_dir.exists() and test_dir.exists() and not force:
        print(f"[data_loader] Data already cached at {DATA_DIR}")
        return train_dir, test_dir

    os.environ["HF_TOKEN"] = HF_TOKEN
    from datasets import load_dataset, Dataset

    print(f"[data_loader] Streaming from {HF_DATASET_REPO}/{HF_DATASET_LANG} ...")
    print(f"[data_loader] Target size: {max_bytes / 1024 / 1024:.0f} MB, "
          f"min_stars: {min_stars}")

    ds = load_dataset(
        HF_DATASET_REPO,
        data_dir=HF_DATASET_LANG,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    # Skip shuffle in quick mode to start collecting immediately
    if not quick:
        ds = ds.shuffle(buffer_size=SAMPLE_SHUFFLE_BUFFER, seed=SAMPLE_SEED)

    collected = []
    total_bytes = 0
    scanned = 0

    for example in tqdm(ds, desc="Streaming"):
        scanned += 1

        stars = _extract_stars(example)
        if stars < min_stars:
            continue

        content = example.get("content", "")
        if not content.strip():
            continue

        content_bytes = len(content.encode("utf-8"))

        if content_bytes < 100 or content_bytes > 500_000:
            continue

        # Basic Python syntax check
        try:
            compile(content, "<string>", "exec")
        except SyntaxError:
            continue

        collected.append({
            "content": content,
            "stars": stars,
            "repo_name": example.get("max_stars_repo_name", ""),
            "file_path": example.get("max_stars_repo_path", ""),
        })
        total_bytes += content_bytes

        if scanned % 500 == 0:
            tqdm.write(f"  scanned={scanned}, collected={len(collected)}, "
                       f"size={total_bytes/1024/1024:.1f}MB")

        if total_bytes >= max_bytes:
            break

    if not collected:
        raise RuntimeError(
            "No examples collected. Check HF token and dataset access."
        )

    print(f"[data_loader] Scanned {scanned} files, "
          f"collected {len(collected)} files, "
          f"total {total_bytes / 1024 / 1024:.1f} MB")

    # Deterministic split
    import random
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(collected)

    split_idx = int(len(collected) * train_ratio)
    train_data = collected[:split_idx]
    test_data = collected[split_idx:]

    print(f"[data_loader] Train: {len(train_data)} files, "
          f"Test: {len(test_data)} files")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    Dataset.from_list(train_data).save_to_disk(str(train_dir))
    Dataset.from_list(test_data).save_to_disk(str(test_dir))

    meta = {
        "total_files": len(collected),
        "total_bytes": total_bytes,
        "train_files": len(train_data),
        "test_files": len(test_data),
        "min_stars": min_stars,
        "max_bytes": max_bytes,
        "scanned": scanned,
    }
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[data_loader] Saved to {DATA_DIR}")
    return train_dir, test_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stream_and_save(force=args.force, quick=args.quick)
