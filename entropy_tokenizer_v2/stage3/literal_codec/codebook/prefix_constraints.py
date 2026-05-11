"""Prefix and reserved-value constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

from .trie import PrefixFreeTrie


@dataclass(slots=True)
class PrefixConstraintChecker:
    """
    Check if a code is valid under:
    - prefix-free code set
    - reserved strings
    - escape prefix collision
    """

    escape_prefix: str
    reserved_strings: set[str] = field(default_factory=set)
    trie: PrefixFreeTrie = field(default_factory=PrefixFreeTrie)

    def is_feasible(self, code: str) -> bool:
        if not code:
            return False
        if code.startswith(self.escape_prefix):
            return False
        if code in self.reserved_strings:
            return False
        return self.trie.can_insert(code)

    def try_add(self, code: str) -> bool:
        if not self.is_feasible(code):
            return False
        return self.trie.insert(code)
