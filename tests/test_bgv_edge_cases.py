"""Boundary and invalid-input BGV checks."""

from __future__ import annotations

import pytest

from diffusion_hash_inv.encoding.bgv import BGVDecoder, BGVEncoder


@pytest.mark.parametrize(
    "message",
    (
        b"\x00" * 4,
        b"\xff" * 31,
        b"\x00\xff" * 15 + b"\x00",
        b"Z" * 31,
    ),
)
def test_zero_ff_alternating_and_repeated_bytes_round_trip(message: bytes) -> None:
    result = BGVDecoder().decode(BGVEncoder().encode(message))
    assert result.valid and result.message == message


def test_invalid_length_and_length_slot_are_rejected() -> None:
    encoder, decoder = BGVEncoder(), BGVDecoder()
    with pytest.raises(ValueError):
        encoder.encode(b"abc")
    image = encoder.encode(b"HELLO")
    image[1, :8, :16] = 0
    assert decoder.decode(image).reason == "length_slot_invalid"
