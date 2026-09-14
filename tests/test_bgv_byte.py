"""Byte-glyph encoding checks."""

from __future__ import annotations

from diffusion_hash_inv.encoding.bgv import byte_to_bits, decode_byte_glyph, encode_byte_glyph


def test_all_256_byte_values() -> None:
    for value in range(256):
        assert decode_byte_glyph(encode_byte_glyph(value)) == value


def test_bits_are_msb_first() -> None:
    assert byte_to_bits(0x4C) == (0, 1, 0, 0, 1, 1, 0, 0)
