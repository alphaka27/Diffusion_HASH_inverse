"""Selectable character groups for random message generation."""

from __future__ import annotations

from string import ascii_lowercase, ascii_uppercase, digits, punctuation


CHARACTER_GROUPS = {
    "lowercase": ascii_lowercase,
    "uppercase": ascii_uppercase,
    "digits": digits,
    "punctuation": punctuation,
}


def select_characters(*groups: str) -> str:
    """Return the combined candidate characters for the named groups."""
    if not groups:
        raise ValueError("at least one character group is required")
    try:
        return "".join(CHARACTER_GROUPS[group] for group in groups)
    except KeyError as error:
        raise ValueError(f"unsupported character group: {error.args[0]!r}") from error


__all__ = ["CHARACTER_GROUPS", "select_characters"]
