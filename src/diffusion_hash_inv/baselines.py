"""Fixed-budget source-prior random-search baseline."""

from __future__ import annotations

import random
import hashlib
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
    attempts = []
    for target in targets:
        if length_aware and not min_length <= len(target.message) <= max_length:
            raise ValueError("known length outside source range")
        stream_key = f"{seed}:{target.source}:{target.algorithm}:{target.q}:{target.prefix}"
        if length_aware:
            stream_key += f":{len(target.message)}"
        stream_seed = int.from_bytes(hashlib.sha256(stream_key.encode()).digest()[:8], "big")
        rng = random.Random(stream_seed)
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
                    generation_seed=stream_seed,
                )
                for _ in range(k)
            )
        )
    return tuple(attempts)


def source_prior_expectation(targets, *, draws: int, seed: int, ks=(1, 10, 100),
                             min_length: int, max_length: int, length_aware: bool):
    """Independent, explicitly budgeted Monte Carlo under the actual source law.

    Estimates are diagnostics, never substituted for realized paired baseline outcomes.
    """
    from .evaluation import binomial_ci95
    from .dataset import digest_prefix_hex
    if draws < 1 or not targets or any(k < 1 for k in ks):
        raise ValueError("positive draw count, budgets and targets required")
    attempts = source_prior_random_search(targets, k=draws, seed=seed, length_aware=length_aware,
                                         min_length=min_length, max_length=max_length)
    rows = []
    for target, group in zip(targets, attempts):
        hits = sum(digest_prefix_hex(hashlib.new(target.algorithm, attempt.message).digest(), target.q) == target.prefix
                   for attempt in group)
        interval = binomial_ci95(hits, draws)
        rows.append({"target_prefix": target.prefix, "draws": draws, "hits": hits, "single_draw_estimate": hits / draws,
                     "single_draw_ci95": interval, "at_k_estimate": {str(k): 1 - (1 - hits / draws)**k for k in ks},
                     "at_k_ci95": {str(k): [1 - (1 - p)**k for p in interval] for k in ks},
                     "zero_hits_is_not_zero_probability": hits == 0})
    return rows


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
