"""Candidate short-code pool generator."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from ..config import CandidateSearchConfig
from ..tokenizer.base import TokenizerAdapter


@dataclass(slots=True, frozen=True)
class CandidateCode:
    """One candidate code with tokenizer-aware cost."""

    code: str
    token_cost: int


class CandidatePoolGenerator:
    """
    Best-first candidate enumeration with a heap.
    Avoid generating full Cartesian product upfront.
    """

    def __init__(self, tokenizer: TokenizerAdapter, config: CandidateSearchConfig) -> None:
        self._tokenizer = tokenizer
        self._config = config

    def generate(self, needed: int, escape_prefix: str, reserved_strings: set[str]) -> list[CandidateCode]:
        if needed <= 0:
            return []

        alphabet = sorted(set(self._config.alphabet))
        if not alphabet:
            return []

        target = max(needed * self._config.oversubscribe_factor, needed)
        heap: list[tuple[int, int, str]] = []
        visited: set[str] = set()
        result: list[CandidateCode] = []

        for ch in alphabet:
            cost = self._tokenizer.token_length(ch)
            heapq.heappush(heap, (cost, 1, ch))
            visited.add(ch)

        expanded = 0
        while heap and len(result) < target and expanded < self._config.max_nodes_to_expand:
            token_cost, char_len, code = heapq.heappop(heap)
            expanded += 1

            if (
                code not in reserved_strings
                and not code.startswith(escape_prefix)
                and code not in self._config.reserved_strings
            ):
                result.append(CandidateCode(code=code, token_cost=token_cost))

            if char_len >= self._config.max_code_length_chars:
                continue

            for ch in alphabet:
                nxt = f"{code}{ch}"
                if nxt in visited:
                    continue
                visited.add(nxt)
                nxt_cost = self._tokenizer.token_length(nxt)
                heapq.heappush(heap, (nxt_cost, char_len + 1, nxt))

        result.sort(key=lambda item: (item.token_cost, len(item.code), item.code))
        return result
