"""Character Glyph Grid Encoding (CGGE) for Printable ASCII messages.

The glyphs are embedded here, rather than rasterised at runtime, so the
representation is identical on every host.  They are the public-domain IBM
VGA 8x8 glyphs distributed by the font8x8 project.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from .bgv import DecodeResult


PRINTABLE94 = "".join(chr(code) for code in range(0x21, 0x7F))
GLYPH_TABLE_VERSION = "font8x8-basic-v1"

# One eight-byte, least-significant-bit-first glyph for each character in
# PRINTABLE94.  Source: public-domain IBM VGA glyphs (font8x8_basic.h).
_GLYPH_BYTES = bytes.fromhex(
    """
    183c3c1818001800 3636000000000000 36367f367f363600 0c3e031e301f0c00
    006333180c666300 1c361c6e3b336e00 0606030000000000 180c0606060c1800
    060c1818180c0600 00663cff3c660000 000c0c3f0c0c0000 00000000000c0c06
    0000003f00000000 00000000000c0c00 6030180c06030100 3e63737b6f673e00
    0c0e0c0c0c0c3f00 1e33301c06333f00 1e33301c30331e00 383c36337f307800
    3f031f3030331e00 1c06031f33331e00 3f3330180c0c0c00 1e33331e33331e00
    1e33333e30180e00 000c0c00000c0c00 000c0c00000c0c06 180c0603060c1800
    00003f00003f0000 060c1830180c0600 1e3330180c000c00 3e637b7b7b031e00
    0c1e33333f333300 3f66663e66663f00 3c66030303663c00 1f36666666361f00
    7f46161e16467f00 7f46161e16060f00 3c66030373667c00 3333333f33333300
    1e0c0c0c0c0c1e00 7830303033331e00 6766361e36666700 0f06060646667f00
    63777f7f6b636300 63676f7b73636300 1c36636363361c00 3f66663e06060f00
    1e3333333b1e3800 3f66663e36666700 1e33070e38331e00 3f2d0c0c0c0c1e00
    3333333333333f00 33333333331e0c00 6363636b7f776300 63631e1c1c366300
    3333331e0c0c1e00 7f6331184c667f00 1e06060606061e00 03060c1830604000
    1e18181818181e00 081c366300000000 00000000000000ff 0c0c180000000000
    00001e303e336e00 0706063e66663b00 00001e3303331e00 3830303e33336e00
    00001e333f031e00 1c36060f06060f00 00006e33333e301f 0706366e66666700
    0c000e0c0c0c1e00 300030303033331e 070666361e366700 0e0c0c0c0c0c1e00
    0000337f7f6b6300 00001f3333333300 00001e3333331e00 00003b66663e060f
    00006e33333e3078 00003b6e66060f00 00003e031e301f00 080c3e0c0c2c1800
    0000333333336e00 00003333331e0c00 0000636b7f7f3600 000063361c366300
    00003333333e301f 00003f190c263f00 380c0c070c0c3800 1818180018181800
    070c0c380c0c0700 6e3b000000000000
    """
)
if len(_GLYPH_BYTES) != len(PRINTABLE94) * 8:  # pragma: no cover - module invariant
    raise RuntimeError("CGGE glyph table must contain exactly 94 8x8 glyphs")


def glyph_table_checksum() -> str:
    """Return the manifest checksum for the embedded, fixed glyph table."""
    return hashlib.sha256(_GLYPH_BYTES).hexdigest()


def _character(value: str | int) -> str:
    if isinstance(value, int):
        if 0x21 <= value <= 0x7E:
            return chr(value)
    elif isinstance(value, str) and len(value) == 1 and value in PRINTABLE94:
        return value
    raise ValueError("character must be one Printable ASCII character (0x21--0x7e)")


def glyph_for_character(value: str | int) -> Tensor:
    """Return one canonical 8x8 glyph as a float tensor in [0, 1]."""
    index = ord(_character(value)) - 0x21
    rows = _GLYPH_BYTES[index * 8 : (index + 1) * 8]
    return torch.tensor([[(row >> column) & 1 for column in range(8)] for row in rows], dtype=torch.float32)


_PROTOTYPES = torch.stack([glyph_for_character(character) for character in PRINTABLE94])


@dataclass(frozen=True)
class CGGEConfig:
    min_message_length: int = 4
    max_message_length: int = 31
    rows: int | None = None
    cols: int = 8
    glyph_size: int = 8
    mask_threshold: float = 0.5
    glyph_valid_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.rows is None:
            object.__setattr__(self, "rows", (self.max_message_length + 8) // 8)
        if self.min_message_length < 0 or self.min_message_length > self.max_message_length:
            raise ValueError("invalid message-length range")
        if self.cols != 8 or self.rows != (self.max_message_length + 8) // 8:
            raise ValueError("grid must be the minimal eight-column grid")
        if self.glyph_size != 8:
            raise ValueError("the fixed glyph table is 8x8")
        if not 0 <= self.mask_threshold <= 1 or not 0 <= self.glyph_valid_threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")

    @property
    def image_height(self) -> int:
        return self.rows * self.glyph_size

    @property
    def image_width(self) -> int:
        return self.cols * self.glyph_size

    @property
    def slot_count(self) -> int:
        return self.rows * self.cols


def _config(config: CGGEConfig | None) -> CGGEConfig:
    return CGGEConfig() if config is None else config


def _printable_bytes(message: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(message, str):
        try:
            value = message.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("message must contain only Printable ASCII characters") from error
    elif isinstance(message, (bytes, bytearray, memoryview)):
        value = bytes(message)
    else:
        raise TypeError("message must be str or bytes-like")
    if any(byte < 0x21 or byte > 0x7E for byte in value):
        raise ValueError("message must contain only Printable ASCII characters")
    return value


class CGGEEncoder:
    """Encode Printable ASCII payloads as canonical [glyph, mask] tensors."""

    def __init__(self, config: CGGEConfig | None = None) -> None:
        self.config = _config(config)

    def encode(self, message: str | bytes | bytearray | memoryview) -> Tensor:
        payload = _printable_bytes(message)
        if not self.config.min_message_length <= len(payload) <= self.config.max_message_length:
            raise ValueError(
                f"message length must be in [{self.config.min_message_length}, {self.config.max_message_length}]"
            )
        image = torch.zeros((2, self.config.image_height, self.config.image_width), dtype=torch.float32)
        for slot, byte in enumerate(payload):
            row, column = divmod(slot, self.config.cols)
            top, left = row * 8, column * 8
            image[0, top : top + 8, left : left + 8] = glyph_for_character(byte)
            image[1, top : top + 8, left : left + 8] = 1
        return image


class CGGEDecoder:
    """Decode CGGE tensors with strict contiguous masks and nearest glyphs."""

    def __init__(self, config: CGGEConfig | None = None) -> None:
        self.config = _config(config)

    def _cell(self, channel: Tensor, slot: int) -> Tensor:
        row, column = divmod(slot, self.config.cols)
        top, left = row * 8, column * 8
        return channel[top : top + 8, left : left + 8]

    def decode(self, image: Tensor, *, normalized: bool = False) -> DecodeResult:
        """Decode a [2, 32, 64] tensor; set ``normalized`` for [-1, 1] samples."""
        expected_shape = (2, self.config.image_height, self.config.image_width)
        if not isinstance(image, Tensor):
            raise TypeError("image must be a torch.Tensor")
        if tuple(image.shape) != expected_shape:
            return DecodeResult(None, False, None, f"invalid_shape_expected_{expected_shape}")
        if not torch.isfinite(image).all():
            return DecodeResult(None, False, None, "non_finite")
        unit_image = (image + 1.0) / 2.0 if normalized else image
        valid_slots = tuple(
            self._cell(unit_image[1], slot).mean().item() >= self.config.mask_threshold
            for slot in range(self.config.slot_count)
        )
        length = sum(valid_slots[:self.config.max_message_length])
        if any(valid_slots[self.config.max_message_length:]) or not self.config.min_message_length <= length <= self.config.max_message_length:
            return DecodeResult(None, False, length, "mask_inconsistent")
        if any(slot_is_valid != (slot < length) for slot, slot_is_valid in enumerate(valid_slots)):
            return DecodeResult(None, False, length, "mask_inconsistent")

        message = bytearray()
        for slot in range(length):
            distances = ((_PROTOTYPES.to(unit_image) - self._cell(unit_image[0], slot)) ** 2).mean(dim=(1, 2))
            distance, index = distances.min(dim=0)
            if distance.item() > self.config.glyph_valid_threshold:
                return DecodeResult(None, False, length, "glyph_too_distant")
            message.append(ord(PRINTABLE94[index.item()]))
        return DecodeResult(bytes(message), True, length, None)


__all__ = [
    "CGGEConfig",
    "CGGEDecoder",
    "CGGEEncoder",
    "GLYPH_TABLE_VERSION",
    "PRINTABLE94",
    "glyph_for_character",
    "glyph_table_checksum",
]
