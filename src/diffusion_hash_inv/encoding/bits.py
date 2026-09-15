"""Direct binary record representation used as the non-image baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .bgv import DecodeResult, bits_to_byte, byte_to_bits


@dataclass(frozen=True)
class DirectBitsConfig:
    min_message_length: int = 4
    max_message_length: int = 31
    bit_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.min_message_length < 0 or self.min_message_length > self.max_message_length:
            raise ValueError("invalid message-length range")
        if not 0 <= self.bit_threshold <= 1:
            raise ValueError("bit_threshold must be in [0, 1]")

    @property
    def slot_count(self) -> int:
        return self.max_message_length + 1

    @property
    def record_bits(self) -> int:
        return self.slot_count * 8


def _config(config: DirectBitsConfig | None) -> DirectBitsConfig:
    return DirectBitsConfig() if config is None else config


class DirectBitsEncoder:
    """Encode a one-byte length header and up to 31 payload bytes as [32, 8]."""

    def __init__(self, config: DirectBitsConfig | None = None) -> None:
        self.config = _config(config)

    def encode(self, message: bytes | bytearray | memoryview) -> Tensor:
        if not isinstance(message, (bytes, bytearray, memoryview)):
            raise TypeError("message must be bytes-like")
        payload = bytes(message)
        if not self.config.min_message_length <= len(payload) <= self.config.max_message_length:
            raise ValueError(
                f"message length must be in [{self.config.min_message_length}, {self.config.max_message_length}]"
            )
        record = torch.zeros((self.config.slot_count, 8), dtype=torch.float32)
        for slot, byte in enumerate((len(payload), *payload)):
            record[slot] = torch.tensor(byte_to_bits(byte), dtype=torch.float32)
        return record


class DirectBitsDecoder:
    """Restore a record from thresholded bits and validate its length header."""

    def __init__(self, config: DirectBitsConfig | None = None) -> None:
        self.config = _config(config)

    def decode(self, record: Tensor, *, normalized: bool = False) -> DecodeResult:
        expected_shape = (self.config.slot_count, 8)
        if not isinstance(record, Tensor):
            raise TypeError("record must be a torch.Tensor")
        if tuple(record.shape) != expected_shape:
            return DecodeResult(None, False, None, f"invalid_shape_expected_{expected_shape}")
        unit_record = (record + 1.0) / 2.0 if normalized else record
        values = [bits_to_byte((row >= self.config.bit_threshold).tolist()) for row in unit_record]
        length = values[0]
        if not self.config.min_message_length <= length <= self.config.max_message_length:
            return DecodeResult(None, False, length, "length_out_of_range")
        return DecodeResult(bytes(values[1 : length + 1]), True, length, None)


__all__ = ["DirectBitsConfig", "DirectBitsDecoder", "DirectBitsEncoder"]
