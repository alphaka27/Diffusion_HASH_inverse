"""Fixed-budget source-prior random-search baseline."""

from __future__ import annotations

import random
from typing import Sequence

from .dataset import DigestRecord
from .encoding.cgge import PRINTABLE94
from .evaluation import CandidateAttempt


def _sample_message(
    rng: random.Random, source: str, *, min_length: int, max_length: int, length: int | None
) -> bytes:
    size = rng.randint(min_length, max_length) if length is None else length
    if source == "printable":
        alphabet = PRINTABLE94.encode("ascii")
        return bytes(alphabet[rng.randrange(len(alphabet))] for _ in range(size))
    if source == "random_bytes":
        return rng.randbytes(size)
    raise ValueError("unsupported source distribution")


def source_prior_random_search(
    targets: Sequence[DigestRecord], *, k: int, seed: int, length_aware: bool = False, min_length: int = 4, max_length: int = 31
) -> tuple[tuple[CandidateAttempt, ...], ...]:
    """Sample exactly k valid source-prior candidates for every target."""
    if k < 1 or min_length < 0 or min_length > max_length:
        raise ValueError("invalid candidate budget or length range")
    rng = random.Random(seed)
    attempts = []
    for target in targets:
        attempts.append(
            tuple(
                CandidateAttempt(
                    _sample_message(
                        rng,
                        target.source,
                        min_length=min_length,
                        max_length=max_length,
                        length=len(target.message) if length_aware else None,
                    ),
                    True,
                )
                for _ in range(k)
            )
        )
    return tuple(attempts)


def nearest_training_digest(
    targets: Sequence[DigestRecord], training_records: Sequence[DigestRecord], *, k: int
) -> tuple[tuple[CandidateAttempt, ...], ...]:
    """Return the closest q-bit training digest record as a leakage diagnostic."""
    if k < 1 or not training_records:
        raise ValueError("candidate budget must be positive and training records non-empty")
    if targets and any(
        (record.algorithm, record.q) != (targets[0].algorithm, targets[0].q) for record in training_records
    ):
        raise ValueError("targets and training records must share algorithm and q")
    attempts = []
    for target in targets:
        target_prefix = int(target.prefix, 16)
        closest = min(
            training_records,
            key=lambda record: ((int(record.prefix, 16) ^ target_prefix).bit_count(), record.id),
        )
        attempts.append(tuple(CandidateAttempt(closest.message, True) for _ in range(k)))
    return tuple(attempts)


__all__ = ["nearest_training_digest", "source_prior_random_search"]
