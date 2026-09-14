"""CGGE fixed-table and strict decoder checks."""

import torch

from diffusion_hash_inv.encoding.cgge import (
    CGGEDecoder,
    CGGEEncoder,
    PRINTABLE94,
    glyph_for_character,
    glyph_table_checksum,
)


def test_all_printable_glyphs_are_unique_and_versionable() -> None:
    glyphs = {bytes(map(int, glyph_for_character(character).flatten().tolist())) for character in PRINTABLE94}
    assert len(glyphs) == 94
    assert len(glyph_table_checksum()) == 64


def test_every_valid_length_round_trips_and_normalized_samples_decode() -> None:
    encoder, decoder = CGGEEncoder(), CGGEDecoder()
    for length in range(4, 32):
        message = (PRINTABLE94 * 2)[:length]
        image = encoder.encode(message)
        decoded = decoder.decode(image)
        assert decoded.valid and decoded.message == message.encode() and decoded.length == length
    assert decoder.decode(encoder.encode("Test") * 2 - 1, normalized=True).message == b"Test"


def test_reserve_or_non_contiguous_masks_are_invalid() -> None:
    image = CGGEEncoder().encode("Test")
    image[1, 24:32, 56:64] = 1  # Reserve cell 31.
    assert CGGEDecoder().decode(image).reason == "mask_inconsistent"
    image = CGGEEncoder().encode("Test")
    image[1, :8, 8:16] = 0  # Slot 1 invalid while later cells remain valid.
    assert CGGEDecoder().decode(image).reason == "mask_inconsistent"


def test_distant_valid_glyph_is_invalid() -> None:
    image = CGGEEncoder().encode("Test")
    image[0, :8, :8] = torch.ones((8, 8))
    assert CGGEDecoder().decode(image).reason == "glyph_too_distant"
