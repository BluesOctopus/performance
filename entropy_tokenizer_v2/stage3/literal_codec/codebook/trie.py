"""Trie for prefix-free code validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    terminal: bool = False


class PrefixFreeTrie:
    """
    Maintain prefix-free constraints.
    Conflict if:
    1) new code is prefix of existing code
    2) existing code is prefix of new code
    """

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def can_insert(self, code: str) -> bool:
        if not code:
            return False
        node = self._root
        for ch in code:
            if node.terminal:
                return False
            if ch not in node.children:
                node = None
                break
            node = node.children[ch]

        if node is not None:
            if node.terminal:
                return False
            if node.children:
                return False
        return True

    def insert(self, code: str) -> bool:
        if not self.can_insert(code):
            return False
        node = self._root
        for ch in code:
            node = node.children.setdefault(ch, _TrieNode())
        node.terminal = True
        self._size += 1
        return True
