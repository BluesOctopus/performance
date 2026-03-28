import re
from pathlib import Path

from spy import Transformer, SPECIAL_TOKENS
import tiktoken


def load_samples(sample_path: Path) -> list[str]:
    text = sample_path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"<\|sample_\d+\|>\n", text)
    return [s for s in parts if s.strip()]


def build_gpt4_spy_tokenizer():
    base_enc = tiktoken.encoding_for_model("gpt-4")
    enc = tiktoken.Encoding(
        name="gpt4-spy",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={
            **base_enc._special_tokens,
            **{v: i + 100264 for i, v in enumerate(SPECIAL_TOKENS)},
        },
    )
    allowed_special_all = set(["<|endoftext|>"] + list(enc._special_tokens.keys()))
    allowed_simpy_special = set(enc._special_tokens.keys())
    return enc, allowed_special_all, allowed_simpy_special


def build_hf_tokenizer(model_name: str, transformer: Transformer):
    # SimPy special tokens should be recognized as single tokens.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.add_special_tokens({"additional_special_tokens": transformer.special_tokens})
    return tok


def main():
    # v2-generated sample file: concatenated with markers like <|sample_0|>
    sample_path = Path(__file__).resolve().parents[1] / "data" / "starcoder_1m_tokens.txt"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    samples = load_samples(sample_path)
    print("Loaded samples:", len(samples))

    transformer = Transformer(ignore_error=True)

    # --- Prepare tokenizers (match v2 intent: compare token count under each tokenizer) ---
    tokenizers = {
        "gpt4": {"type": "tiktoken", "tok_key": "gpt4"},
        "codegen-350M-mono": {"type": "hf", "model_name": "Salesforce/codegen-350M-mono"},
        "santacoder": {"type": "hf", "model_name": "bigcode/santacoder"},
    }

    gpt4_enc, allowed_special_all, allowed_simpy_special = build_gpt4_spy_tokenizer()
    tokenizers["gpt4"]["enc"] = gpt4_enc
    tokenizers["gpt4"]["allowed_special_all"] = allowed_special_all
    tokenizers["gpt4"]["allowed_simpy_special"] = allowed_simpy_special

    # --- Evaluate ---
    for tok_key, meta in tokenizers.items():
        print(f"\n=== SimPy token eval: {tok_key} ===")

        # Build HF tokenizer lazily to avoid network if we only need gpt4.
        if meta["type"] == "hf":
            try:
                tok = build_hf_tokenizer(meta["model_name"], transformer)
            except Exception as e:
                print(f"[skip] failed to load tokenizer {meta['model_name']}: {e}")
                continue
        else:
            tok = meta["enc"]

        filtered_raw = 0
        parsed_total = 0
        parsed_ok = 0
        parsed_fail = 0

        for code in samples:
            # Mimic token_count.py: remove first line if it starts with '<'
            # (marker-like line), but our file is already split; keep this for safety.
            lines = code.splitlines()
            if lines and lines[0].startswith("<"):
                code = "\n".join(lines[1:])

            # Skip shebang-like scripts (token_count.py behavior)
            if code.startswith("#!/"):
                continue

            # Raw token count
            if meta["type"] == "tiktoken":
                raw_len = len(
                    tok.encode(code, allowed_special=meta["allowed_special_all"])
                )
            else:
                raw_len = len(tok.encode(code))
            filtered_raw += raw_len

            # Parse to SimPy code
            try:
                spy_code = transformer.parse(code)
            except Exception:
                parsed_fail += 1
                continue

            parsed_ok += 1

            if meta["type"] == "tiktoken":
                parsed_len = len(
                    tok.encode(spy_code, allowed_special=meta["allowed_simpy_special"])
                )
            else:
                parsed_len = len(tok.encode(spy_code))

            parsed_total += parsed_len

        print("Parsed OK:", parsed_ok, "Parsed failed:", parsed_fail)
        print("Filtered raw tokens:", filtered_raw)
        print("SimPy parsed tokens:", parsed_total)
        if filtered_raw:
            ratio = parsed_total / filtered_raw
            reduction_pct = (1.0 - ratio) * 100.0
            print("Parsed/Filtered ratio:", ratio)
            print("Reduction vs filtered raw (%):", reduction_pct)


if __name__ == "__main__":
    main()

