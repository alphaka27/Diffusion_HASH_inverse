"""Independent wire-format checks for the three planned representations."""

from __future__ import annotations

import torch

from diffusion_hash_inv.encoding import (
    BGVDecoder,
    BGVEncoder,
    CGGEDecoder,
    CGGEEncoder,
    DirectBitsDecoder,
    DirectBitsEncoder,
    PRINTABLE94,
    glyph_for_character,
    glyph_table_checksum,
)


CGGE_GLYPH_TABLE_SHA256 = "6ef6d0bfa6e29c823ad9ff7d6b4b29b9af5f739fee711268ff5b83ca7aeed50a"


def _bits(value: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(7, -1, -1)]


def _bgv_oracle(message: bytes) -> torch.Tensor:
    image = torch.zeros((2, 32, 128), dtype=torch.float32)
    for slot, value in enumerate((len(message), *message)):
        row, column = divmod(slot, 8)
        top, left = row * 8, column * 16
        image[0, top : top + 8, left : left + 16] = torch.tensor(_bits(value), dtype=torch.float32).reshape(2, 4).repeat_interleave(4, 0).repeat_interleave(4, 1)
        image[1, top : top + 8, left : left + 16] = 1
    return image


def _bits_oracle(message: bytes) -> torch.Tensor:
    values = (len(message), *message, *((0,) * (31 - len(message))))
    return torch.tensor([_bits(value) for value in values], dtype=torch.float32)


def _cgge_oracle(message: bytes) -> torch.Tensor:
    image = torch.zeros((2, 32, 64), dtype=torch.float32)
    for slot, value in enumerate(message):
        row, column = divmod(slot, 8)
        top, left = row * 8, column * 8
        image[0, top : top + 8, left : left + 8] = glyph_for_character(value)
        image[1, top : top + 8, left : left + 8] = 1
    return image


def _messages() -> tuple[bytes, ...]:
    return tuple(bytes((start + index) % 256 for index in range(length)) for length in range(4, 32) for start in (0, 127, 255))


def test_bgv_and_direct_bits_match_independent_byte_layout_oracles() -> None:
    bgv_encoder, bgv_decoder = BGVEncoder(), BGVDecoder()
    bits_encoder, bits_decoder = DirectBitsEncoder(), DirectBitsDecoder()
    for message in _messages():
        bgv = bgv_encoder.encode(message)
        direct = bits_encoder.encode(message)
        assert torch.equal(bgv, _bgv_oracle(message))
        assert torch.equal(direct, _bits_oracle(message))
        assert bgv_decoder.decode(bgv).message == message
        assert bgv_decoder.decode(bgv * 2 - 1, normalized=True).message == message
        assert bits_decoder.decode(direct).message == message
        assert bits_decoder.decode(direct * 2 - 1, normalized=True).message == message


def test_cgge_matches_fixed_table_and_independent_grid_layout() -> None:
    assert glyph_table_checksum() == CGGE_GLYPH_TABLE_SHA256
    assert len({bytes(glyph_for_character(character).to(torch.uint8).flatten().tolist()) for character in PRINTABLE94}) == 94
    encoder, decoder = CGGEEncoder(), CGGEDecoder()
    messages = tuple(character.encode() * 4 for character in PRINTABLE94) + tuple(
        (PRINTABLE94 * 2)[start : start + length].encode() for length in range(4, 32) for start in range(0, 94, 7)
    )
    for message in messages:
        image = encoder.encode(message)
        assert torch.equal(image, _cgge_oracle(message))
        assert decoder.decode(image).message == message
        assert decoder.decode(image * 2 - 1, normalized=True).message == message
