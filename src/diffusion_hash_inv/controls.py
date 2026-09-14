"""Representation controls sharing the normal decoder and hash evaluator."""

from __future__ import annotations

import random
from typing import Literal, Sequence

from .dataset import DigestRecord
from .encoding import BGVDecoder, BGVEncoder, CGGEDecoder, CGGEEncoder
from .evaluation import CandidateAttempt


Representation = Literal["bgv", "cgge"]


def _round_trip(message: bytes, representation: Representation) -> CandidateAttempt:
    if representation == "bgv":
        decoded = BGVDecoder().decode(BGVEncoder().encode(message))
    elif representation == "cgge":
        decoded = CGGEDecoder().decode(CGGEEncoder().encode(message))
    else:  # pragma: no cover - Literal plus public input validation
        raise ValueError("representation must be bgv or cgge")
    return CandidateAttempt(decoded.message, decoded.valid, decoded.reason)


def representation_control(
    targets: Sequence[DigestRecord], *, representation: Representation, kind: Literal["reversible", "shuffled", "zero"], k: int, seed: int = 0
) -> tuple[tuple[CandidateAttempt, ...], ...]:
    """Produce k decoded control candidates per target without post-processing.

    ``reversible`` returns each condition's canonical record. ``shuffled``
    pairs every condition with a different record; ``zero`` uses one constant
    record for every condition. All three still use the normal encoder/decoder.
    """
    if k < 1 or not targets:
        raise ValueError("targets must be non-empty and k must be positive")
    if kind == "shuffled" and len(targets) < 2:
        raise ValueError("shuffled control requires at least two targets")
    if representation == "cgge" and any(target.source != "printable" for target in targets):
        raise ValueError("CGGE controls require Printable source records")
    if kind == "reversible":
        sources = tuple(target.message for target in targets)
    elif kind == "zero":
        sources = (targets[0].message,) * len(targets)
    elif kind == "shuffled":
        offset = random.Random(seed).randrange(1, len(targets))
        sources = tuple(targets[(index + offset) % len(targets)].message for index in range(len(targets)))
    else:
        raise ValueError("kind must be reversible, shuffled, or zero")
    return tuple(tuple(_round_trip(source, representation) for _ in range(k)) for source in sources)


__all__ = ["representation_control"]
