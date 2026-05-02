import re


class SimpleOfflineTokenizer:
    """Deterministic local tokenizer for offline tests."""

    def encode(self, text: str, add_special_tokens: bool = False, allowed_special: str = "all"):
        del add_special_tokens, allowed_special
        # Split into words and punctuation-like tokens, deterministic and local.
        return re.findall(r"[A-Za-z_]\w*|\d+|<[^>\s]+>|[^\w\s]", text)
