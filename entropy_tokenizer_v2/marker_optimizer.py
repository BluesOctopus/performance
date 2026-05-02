from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenizer_utils import count_tokens, normalize_tokenizer_name


@dataclass(frozen=True)
class MarkerScheme:
    tokenizer_name: str
    namespace: str
    markers: list[str]
    marker_token_costs: list[int]

    def marker(self, index: int) -> str:
        return self.markers[index]

    def marker_cost(self, index: int) -> int:
        return self.marker_token_costs[index]

    def index_of(self, marker: str) -> int:
        try:
            return self.markers.index(marker)
        except ValueError as exc:
            raise KeyError(marker) from exc


def build_marker_scheme(
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    namespace: str,
    count: int,
    avoid_texts: list[str] | None = None,
) -> MarkerScheme:
    normalized_name = normalize_tokenizer_name(tokenizer_name)
    avoid_texts = list(avoid_texts or [])
    families = _candidate_families(namespace)
    viable: list[tuple[tuple[int, int, int], str, list[str], list[int]]] = []
    fallback_markers = [f"<SYN_{index}>" for index in range(count)]
    fallback_costs = [_token_cost(marker, encoder, tok_type) for marker in fallback_markers]

    for family_name, family_factory, risk_score in families:
        markers = [family_factory(index) for index in range(count)]
        if not _markers_are_viable(markers, avoid_texts):
            continue
        costs = [_token_cost(marker, encoder, tok_type) for marker in markers]
        single_token_penalty = sum(1 for cost in costs if cost != 1)
        total_length = sum(len(marker) for marker in markers)
        rank = (single_token_penalty, sum(costs), total_length + risk_score)
        viable.append((rank, family_name, markers, costs))

    if viable:
        viable.sort(key=lambda item: (item[0], item[1]))
        _, family_name, markers, costs = viable[0]
        return MarkerScheme(
            tokenizer_name=normalized_name,
            namespace=family_name,
            markers=markers,
            marker_token_costs=costs,
        )

    return MarkerScheme(
        tokenizer_name=normalized_name,
        namespace="legacy",
        markers=fallback_markers,
        marker_token_costs=fallback_costs,
    )


def build_legacy_marker_scheme(
    tokenizer_name: str,
    encoder: Any,
    tok_type: str,
    count: int,
) -> MarkerScheme:
    markers = [f"<SYN_{index}>" for index in range(count)]
    return MarkerScheme(
        tokenizer_name=normalize_tokenizer_name(tokenizer_name),
        namespace="legacy",
        markers=markers,
        marker_token_costs=[_token_cost(marker, encoder, tok_type) for marker in markers],
    )


def _candidate_families(namespace: str) -> list[tuple[str, Any, int]]:
    ns = _normalize_namespace(namespace)
    ascii_prefixes = ("@", "#", "$", "%", "&", "!", "?", "~")
    unicode_prefixes = ("§", "¤", "¦", "¶", "※", "◇", "○", "◊")
    families: list[tuple[str, Any, int]] = []

    for prefix in ascii_prefixes:
        families.append((f"ascii_{prefix}", lambda index, prefix=prefix: f"{prefix}{index}", 2))
    for prefix in unicode_prefixes:
        families.append((f"unicode_{prefix}", lambda index, prefix=prefix: f"{prefix}{index}", 1))

    for prefix in ("s", "m", "k", "x"):
        families.append((f"short_{prefix}", lambda index, prefix=prefix: f"{prefix}{index}", 3))
        families.append((f"short_wrapped_{prefix}", lambda index, prefix=prefix: f"{prefix}{index}:", 4))

    families.extend(
        [
            ("ns_ascii", lambda index: f"{ns}{index}", 5),
            ("ns_wrapped", lambda index: f"{ns}_{index}", 6),
            ("ns_angle", lambda index: f"<{ns}{index}>", 7),
            ("legacy", lambda index: f"<SYN_{index}>", 100),
        ]
    )
    return families


def _normalize_namespace(namespace: str) -> str:
    raw = "".join(ch for ch in namespace if ch.isalnum()).lower()
    return raw[:6] or "syn"


def _markers_are_viable(markers: list[str], avoid_texts: list[str]) -> bool:
    if len(set(markers)) != len(markers):
        return False
    if any(not marker.strip() for marker in markers):
        return False
    if any(any(marker in text for text in avoid_texts) for marker in markers):
        return False
    return True


def _token_cost(text: str, encoder: Any, tok_type: str) -> int:
    return count_tokens(text, encoder=encoder, tok_type=tok_type)
