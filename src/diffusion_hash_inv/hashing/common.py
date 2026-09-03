"""Shared 32-bit operations for hash implementations."""

_MASK_32 = 0xFFFFFFFF


def _word(value: int) -> str:
    """Represent a 32-bit word in a JSON-friendly, unambiguous form."""
    return f"0x{value & _MASK_32:08x}"


def _rotate_left(value: int, amount: int) -> int:
    return ((value << amount) | (value >> (32 - amount))) & _MASK_32


def _rotate_right(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (32 - amount))) & _MASK_32
