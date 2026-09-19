"""Byte Glyph Visualization (BGV), a lossless two-channel byte encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class BGVConfig:
    min_message_length: int = 4
    max_message_length: int = 31
    rows: int | None = None
    cols: int = 8
    glyph_rows: int = 2
    glyph_cols: int = 4
    bit_block_size: int = 4
    bit_threshold: float = 0.5
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.rows is None:
            object.__setattr__(self, "rows", (self.max_message_length + 8) // 8)
        if self.min_message_length < 0 or self.max_message_length > 255:
            raise ValueError("message lengths must fit in one byte")
        if self.min_message_length > self.max_message_length:
            raise ValueError("min_message_length must not exceed max_message_length")
        if min(self.rows, self.cols, self.glyph_rows, self.glyph_cols, self.bit_block_size) <= 0:
            raise ValueError("grid dimensions and bit_block_size must be positive")
        if self.glyph_rows * self.glyph_cols != 8:
            raise ValueError("a byte glyph must contain exactly eight bits")
        if self.cols != 8 or self.rows != (self.max_message_length + 8) // 8:
            raise ValueError("slot grid must be the minimal eight-column grid")
        if not 0 <= self.bit_threshold <= 1 or not 0 <= self.mask_threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")

    @property
    def cell_height(self) -> int:
        return self.glyph_rows * self.bit_block_size

    @property
    def cell_width(self) -> int:
        return self.glyph_cols * self.bit_block_size

    @property
    def image_height(self) -> int:
        return self.rows * self.cell_height

    @property
    def image_width(self) -> int:
        return self.cols * self.cell_width

    @property
    def slot_count(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class DecodeResult:
    message: bytes | None
    valid: bool
    length: int | None
    reason: str | None


def _config(config: BGVConfig | None) -> BGVConfig:
    return BGVConfig() if config is None else config


def _validate_byte(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFF:
        raise ValueError("byte value must be an integer in [0, 255]")


def byte_to_bits(value: int) -> tuple[int, ...]:
    """Return the byte's fixed MSB-first bit order."""
    _validate_byte(value)
    return tuple((value >> shift) & 1 for shift in range(7, -1, -1))


def bits_to_byte(bits: Sequence[int | bool]) -> int:
    """Restore a byte from eight fixed-order bits."""
    if len(bits) != 8 or any(bit not in (0, 1, False, True) for bit in bits):
        raise ValueError("bits must contain exactly eight 0/1 values")
    return sum(int(bit) << shift for bit, shift in zip(bits, range(7, -1, -1)))


def encode_byte_glyph(value: int, config: BGVConfig | None = None) -> Tensor:
    """Expand one byte into its 2x4, block-filled glyph."""
    settings = _config(config)
    logical_glyph = torch.tensor(byte_to_bits(value), dtype=torch.float32).reshape(
        settings.glyph_rows, settings.glyph_cols
    )
    return logical_glyph.repeat_interleave(settings.bit_block_size, 0).repeat_interleave(
        settings.bit_block_size, 1
    )


def decode_byte_glyph(glyph: Tensor, config: BGVConfig | None = None) -> int:
    """Threshold block averages to restore one byte glyph."""
    settings = _config(config)
    if not isinstance(glyph, Tensor) or tuple(glyph.shape) != (settings.cell_height, settings.cell_width):
        raise ValueError(f"glyph must have shape ({settings.cell_height}, {settings.cell_width})")
    scores = glyph.reshape(
        settings.glyph_rows,
        settings.bit_block_size,
        settings.glyph_cols,
        settings.bit_block_size,
    ).mean(dim=(1, 3))
    return bits_to_byte((scores >= settings.bit_threshold).flatten().tolist())


class BGVEncoder:
    """Encode 4--31 byte payloads as canonical [glyph, validity-mask] tensors."""

    def __init__(self, config: BGVConfig | None = None) -> None:
        self.config = _config(config)

    def encode(self, message: bytes | bytearray | memoryview) -> Tensor:
        if not isinstance(message, (bytes, bytearray, memoryview)):
            raise TypeError("message must be bytes-like")
        payload = bytes(message)
        if not self.config.min_message_length <= len(payload) <= self.config.max_message_length:
            raise ValueError(
                f"message length must be in [{self.config.min_message_length}, {self.config.max_message_length}]"
            )

        image = torch.zeros((2, self.config.image_height, self.config.image_width), dtype=torch.float32)
        for slot, value in enumerate((len(payload), *payload)):
            row, col = divmod(slot, self.config.cols)
            top, left = row * self.config.cell_height, col * self.config.cell_width
            image[0, top : top + self.config.cell_height, left : left + self.config.cell_width] = encode_byte_glyph(
                value, self.config
            )
            image[1, top : top + self.config.cell_height, left : left + self.config.cell_width] = 1
        return image


class BGVDecoder:
    """Restore BGV tensors and reject malformed length or validity masks."""

    def __init__(self, config: BGVConfig | None = None) -> None:
        self.config = _config(config)

    def _cell(self, channel: Tensor, slot: int) -> Tensor:
        row, col = divmod(slot, self.config.cols)
        top, left = row * self.config.cell_height, col * self.config.cell_width
        return channel[top : top + self.config.cell_height, left : left + self.config.cell_width]

    def decode(self, image: Tensor, *, strict_mask: bool = True, normalized: bool = False) -> DecodeResult:
        """Decode a [2, 32, 128] tensor; set ``normalized`` for [-1, 1] samples."""
        if not isinstance(image, Tensor):
            raise TypeError("image must be a torch.Tensor")
        expected_shape = (2, self.config.image_height, self.config.image_width)
        if tuple(image.shape) != expected_shape:
            return DecodeResult(None, False, None, f"invalid_shape_expected_{expected_shape}")
        if not torch.isfinite(image).all():
            return DecodeResult(None, False, None, "non_finite")

        unit_image = (image + 1.0) / 2.0 if normalized else image
        length = decode_byte_glyph(self._cell(unit_image[0], 0), self.config)
        if not self.config.min_message_length <= length <= self.config.max_message_length:
            return DecodeResult(None, False, length, "length_out_of_range")

        valid_slots = tuple(
            self._cell(unit_image[1], slot).mean().item() >= self.config.mask_threshold
            for slot in range(self.config.slot_count)
        )
        if not valid_slots[0]:
            return DecodeResult(None, False, length, "length_slot_invalid")
        if strict_mask and any(slot_is_valid != (slot <= length) for slot, slot_is_valid in enumerate(valid_slots)):
            return DecodeResult(None, False, length, "mask_inconsistent")
        if any(decode_byte_glyph(self._cell(unit_image[0], slot), self.config) != 0
               for slot in range(length + 1, self.config.slot_count)):
            return DecodeResult(None, False, length, "padding_inconsistent")

        message = bytes(decode_byte_glyph(self._cell(unit_image[0], slot), self.config) for slot in range(1, length + 1))
        return DecodeResult(message, True, length, None)


def plot_bgv(
    image: Tensor,
    *,
    channel: int = 0,
    show_grid: bool = False,
    show_indices: bool = False,
    normalized: bool = False,
    config: BGVConfig | None = None,
):
    """Plot one BGV channel; matplotlib is imported only for this debug helper."""
    settings = _config(config)
    if channel not in (0, 1) or not isinstance(image, Tensor) or tuple(image.shape) != (2, settings.image_height, settings.image_width):
        raise ValueError("image must be a BGV tensor and channel must be 0 or 1")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional debug dependency
        raise RuntimeError("plot_bgv requires matplotlib") from error

    display = (image + 1.0) / 2.0 if normalized else image
    figure, axis = plt.subplots()
    axis.imshow(display[channel].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if show_grid:
        for row in range(settings.rows + 1):
            axis.axhline(row * settings.cell_height - 0.5, color="tab:red", linewidth=0.5)
        for col in range(settings.cols + 1):
            axis.axvline(col * settings.cell_width - 0.5, color="tab:red", linewidth=0.5)
    if show_indices:
        for slot in range(settings.slot_count):
            row, col = divmod(slot, settings.cols)
            axis.text(col * settings.cell_width + 1, row * settings.cell_height + 1, str(slot), color="tab:red")
    axis.set_title(("Byte glyph" if channel == 0 else "Validity mask") + " channel")
    return figure


def save_bgv_image(image: Tensor, path: str | Path, **plot_options: object) -> None:
    """Save a debug plot of either BGV channel."""
    figure = plot_bgv(image, **plot_options)
    figure.savefig(path, bbox_inches="tight")
    figure.clear()
