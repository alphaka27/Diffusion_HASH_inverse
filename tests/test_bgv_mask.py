"""Validity-mask checks."""

from __future__ import annotations

from diffusion_hash_inv.encoding.bgv import BGVDecoder, BGVEncoder


def test_mask_marks_length_and_payload_but_not_padding() -> None:
    image = BGVEncoder().encode(b"HELLO")
    assert image[1, :8, :96].eq(1).all()  # slots 0--5
    assert image[1, :8, 96:].eq(0).all()  # slots 6--7


def test_strict_decoder_rejects_inconsistent_mask() -> None:
    image = BGVEncoder().encode(b"HELLO")
    image[1, :8, 96:112] = 1  # Mark padding slot 6 as valid.
    assert BGVDecoder().decode(image).reason == "mask_inconsistent"
    assert BGVDecoder().decode(image, strict_mask=False).message == b"HELLO"
