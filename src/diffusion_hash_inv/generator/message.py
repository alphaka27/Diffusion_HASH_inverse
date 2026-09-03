"""Generate messages from caller-supplied character candidates."""

from __future__ import annotations

import random
import secrets


def generate_message(
    length: int,
    characters: str,
    *,
    seed: int | float | str | bytes | bytearray | None = None,
) -> str:
    """Return a message of ``length`` from the supplied ``characters``."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative integer")
    if not isinstance(characters, str) or not characters:
        raise ValueError("characters must be a non-empty string")

    choose = secrets.choice if seed is None else random.Random(seed).choice
    return "".join(choose(characters) for _ in range(length))


__all__ = ["generate_message"]
