"""Exhaustive 1--2 byte verifier checks, kept separate from model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

from .dataset import DigestRecord
from .encoding.cgge import PRINTABLE94
from .evaluation import verify_candidate


@dataclass(frozen=True)
class ExhaustiveResult:
    target_id: int
    candidates_examined: int
    matching_messages: tuple[bytes, ...]


def _messages(source: str, max_length: int) -> Iterable[bytes]:
    alphabet = PRINTABLE94.encode("ascii") if source == "printable" else bytes(range(256))
    if source not in {"printable", "random_bytes"}:
        raise ValueError("unsupported source distribution")
    for length in range(1, max_length + 1):
        yield from (bytes(value) for value in product(alphabet, repeat=length))


def exhaustive_sanity_search(targets: Sequence[DigestRecord], *, max_length: int = 2) -> tuple[ExhaustiveResult, ...]:
    """Find all q-bit matches in the declared 1--2 byte source domain."""
    if not 1 <= max_length <= 2:
        raise ValueError("sanity search supports only one- and two-byte domains")
    results = []
    for target in targets:
        matches = []
        examined = 0
        for candidate in _messages(target.source, max_length):
            examined += 1
            if verify_candidate(candidate, target).prefix_match:
                matches.append(candidate)
        results.append(ExhaustiveResult(target.id, examined, tuple(matches)))
    return tuple(results)


__all__ = ["ExhaustiveResult", "exhaustive_sanity_search"]
