"""Lossless categorical records; no target length enters the decoder."""
from dataclasses import dataclass

import torch
from torch import Tensor

from .bgv import DecodeResult

TOKENIZER_VERSION = "categorical-eos-pad-mask-v1"


@dataclass(frozen=True)
class TokenCodec:
    source: str
    max_length: int
    min_length: int = 4

    def __post_init__(self):
        if self.source not in {"printable", "random_bytes"} or not 0 <= self.min_length <= self.max_length:
            raise ValueError("invalid source or length range")

    @property
    def payload_size(self):
        return 94 if self.source == "printable" else 256

    @property
    def eos(self):
        return self.payload_size

    @property
    def pad(self):
        return self.payload_size + 1

    @property
    def mask(self):
        return self.payload_size + 2

    @property
    def vocabulary_size(self):
        return self.payload_size + 3

    def encode(self, message: bytes) -> Tensor:
        if not isinstance(message, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if not self.min_length <= len(message) <= self.max_length:
            raise ValueError("length_out_of_range")
        offset = 0x21 if self.source == "printable" else 0
        payload = [value - offset for value in message]
        if any(not 0 <= value < self.payload_size for value in payload):
            raise ValueError("payload_out_of_domain")
        return torch.tensor(payload + [self.eos] + [self.pad] * (self.max_length - len(message)), dtype=torch.long)

    def decode(self, sequence: Tensor) -> DecodeResult:
        if not isinstance(sequence, Tensor) or tuple(sequence.shape) != (self.max_length + 1,):
            return DecodeResult(None, False, None, "invalid_shape")
        if sequence.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            return DecodeResult(None, False, None, "non_integer_token")
        values = sequence.tolist()
        if any(not 0 <= value < self.vocabulary_size for value in values):
            return DecodeResult(None, False, None, "unknown_token")
        if self.mask in values:
            return DecodeResult(None, False, None, "mask_remaining")
        if values.count(self.eos) != 1:
            return DecodeResult(None, False, None, "eos_count")
        length = values.index(self.eos)
        if not self.min_length <= length <= self.max_length:
            return DecodeResult(None, False, length, "length_out_of_range")
        if any(value >= self.payload_size for value in values[:length]):
            return DecodeResult(None, False, length, "invalid_payload")
        if any(value != self.pad for value in values[length + 1:]):
            return DecodeResult(None, False, length, "non_pad_after_eos")
        offset = 0x21 if self.source == "printable" else 0
        return DecodeResult(bytes(value + offset for value in values[:length]), True, length, None)
