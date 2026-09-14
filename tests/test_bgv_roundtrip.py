"""End-to-end BGV round-trip checks."""

from __future__ import annotations

import random

import torch

from diffusion_hash_inv.encoding.bgv import BGVDecoder, BGVEncoder


def test_all_valid_lengths_round_trip_for_printable_and_random_bytes() -> None:
    encoder, decoder = BGVEncoder(), BGVDecoder()
    random_bytes = random.Random(0)
    for length in range(4, 32):
        for message in (b"A" * length, random_bytes.randbytes(length)):
            result = decoder.decode(encoder.encode(message))
            assert result.valid and result.message == message and result.length == length


def test_encoding_is_deterministic_and_normalized_samples_decode() -> None:
    image = BGVEncoder().encode(b"HELLO")
    assert torch.equal(image, BGVEncoder().encode(b"HELLO"))
    result = BGVDecoder().decode(image * 2 - 1, normalized=True)
    assert result.valid and result.message == b"HELLO"
