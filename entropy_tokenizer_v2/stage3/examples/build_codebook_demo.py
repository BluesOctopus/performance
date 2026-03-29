"""Demo script for offline codebook build and round-trip."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from literal_codec.config import CompressionConfig
    from literal_codec.pipeline.offline_builder import OfflineCodebookBuilder
    from literal_codec.tokenizer.mock_tokenizer import MockTokenizerAdapter

    csv_path = root / "examples" / "sample_literals.csv"
    output_dir = root / "artifacts"

    cfg = CompressionConfig(fields=["service_name", "env", "tag_prefix"])
    builder = OfflineCodebookBuilder(config=cfg, tokenizer=MockTokenizerAdapter())
    artifacts = builder.build_from_csv(csv_path, fields=list(cfg.fields))
    artifacts.save(output_dir)

    sample = {
        "service_name": "serve_auth",
        "env": "prod",
        "tag_prefix": "api_v1",
        "other": "keep_raw",
    }
    encoded = artifacts.encode_record(sample)
    decoded = artifacts.decode_record(encoded)
    print("raw    =", sample)
    print("encoded=", encoded)
    print("decoded=", decoded)
    print("summary=", artifacts.report["summary"])


if __name__ == "__main__":
    main()
