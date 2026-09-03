"""Generate random byte sequences."""

from __future__ import annotations

import random
import secrets


def generate_bytes(
    length: int,
    *,
    seed: int | float | str | bytes | bytearray | None = None,
) -> bytes:
    """Return ``length`` random bytes; use ``seed`` only for repeatable output."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative integer")
    return secrets.token_bytes(length) if seed is None else random.Random(seed).randbytes(length)


__all__ = ["generate_bytes"]
