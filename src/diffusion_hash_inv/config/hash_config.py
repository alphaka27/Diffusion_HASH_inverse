"""
Configuration for hash algorithm parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
import sys
from typing import Dict, Optional, Tuple


def _validate_positive_byte_multiple(name: str, value: int) -> None:
    if value <= 0 or value % 8 != 0:
        raise ValueError(f"{name} must be a positive multiple of 8")

@dataclass(frozen=True)
class MD5Constants:
    """
    Constants for MD5 hash function.
    """

    word_size: int = 32
    block_size: int = 512
    byteorder: str = "little"

    s: Tuple[int, ...] = \
        (7, 12, 17, 22) * 4 + \
        (5, 9, 14, 20) * 4 + \
        (4, 11, 16, 23) * 4 + \
        (6, 10, 15, 21) * 4

    k: Tuple[int, ...] = tuple(
        int(abs((2 ** 32) * abs(math.sin(i + 1)))) for i in range(64)
    )

    init_hash: Dict[str, int] = field(init=False, default_factory=lambda: {
        "A": 0x67452301, # 0x01234567 = 1,732,584,193
        "B": 0xEFCDAB89, # 0x89ABCDEF = 2,402,653,031
        "C": 0x98BADCFE, # 0xFEDCBA98 = 4,277,217,208
        "D": 0x10325476, # 0x76543210 = 1,987,136,336
    })

    hierarchy: Tuple[str, str, str] = field(init=False, \
        default=("Step", "Round", "Loop"))

    def __post_init__(self):
        _validate_positive_byte_multiple("word_size", self.word_size)
        _validate_positive_byte_multiple("block_size", self.block_size)
        object.__setattr__(self, "mask", (1 << self.word_size) - 1)

@dataclass(frozen=True)
class SHA256Constants:
    """
    Constants for SHA256 hash function.
    """
    word_size: int = 32
    block_size: int = 512

    byteorder: str = "big"

    hierarchy: Tuple[str, str, str] = field(init=False, \
        default=("Round", "Step", "Loop"))

    def __post_init__(self):
        _validate_positive_byte_multiple("word_size", self.word_size)
        _validate_positive_byte_multiple("block_size", self.block_size)
        object.__setattr__(self, "mask", (1 << self.word_size) - 1)


_CONSTANTS_BY_ALGORITHM: dict[str, type[MD5Constants] | type[SHA256Constants]] = {
    "md5": MD5Constants,
    "sha256": SHA256Constants,
}

@dataclass(frozen=True)
class HashConfig:
    """
    Configuration for hash generation and validation.
    """
    hash_alg: Optional[str] = None
    length: Optional[int] = None

    constants: MD5Constants | SHA256Constants = \
        field(init=False, default=None)

    def __post_init__(self):
        if self.hash_alg is None:
            raise ValueError("hash_alg must be specified")

        if self.length is None:
            raise ValueError("length must be specified")

        hash_alg_lower = self.hash_alg.lower()
        constants_type = _CONSTANTS_BY_ALGORITHM.get(hash_alg_lower)
        if constants_type is None:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_alg}")
        object.__setattr__(self, "constants", constants_type()) # bypass frozen

        _validate_positive_byte_multiple("length", self.length)
        self._constant_value("word_size")
        self._constant_value("block_size")

    def __repr__(self):
        return (
            "HashConfig\n"
            f"  hash_alg: {self.hash_alg},\n"
            f"  length: {self.length},\n"
            f"  byteorder: {self.byteorder},\n"
            f"  word_size: {self.ws_bits},\n"
            f"  block_size: {self.bs_bits},\n"
            f"  mask: 0x{self.mask:X},\n"
            f"  hierarchy: {self.hierarchy}\n")

    @property
    def byteorder(self) -> str:
        """
        Get the byte order configuration.
        """
        byteorder = self._constant_value("byteorder")
        if byteorder not in ("little", "big"):
            raise ValueError("byteorder must be either 'little' or 'big'")

        return str(byteorder)

    @property
    def ws_bits(self) -> int:
        """
        Get the word size configuration.
        """
        return self._positive_multiple_constant("word_size")

    @property
    def bs_bits(self) -> int:
        """
        Get the block size configuration.
        """
        return self._positive_multiple_constant("block_size")

    @property
    def ws_bytes(self) -> int:
        """
        Get the word size in bytes.
        """
        return self.ws_bits // 8

    @property
    def bs_bytes(self) -> int:
        """
        Get the block size in bytes.
        """
        return self.bs_bits // 8

    @property
    def mask(self) -> int:
        """
        Get the mask configuration.
        """
        return self._constant_value("mask")

    @property
    def hierarchy(self) -> Tuple[str, str, str]:
        """
        Get the hierarchy configuration.
        """
        return self._constant_value("hierarchy")

    @property
    def hash_alg_upper(self) -> str:
        """
        Get the hash algorithm name in uppercase.
        """
        if self.hash_alg is None:
            raise ValueError("hash_alg is not set.")
        return self.hash_alg.upper()

    def _constant_value(self, name: str):
        constants = object.__getattribute__(self, "constants")
        if constants is None:
            raise ValueError("Hash algorithm constants are not set.")
        value = getattr(constants, name)
        if value is None:
            raise ValueError(f"{name} is not set in hash algorithm constants.")
        return value

    def _positive_multiple_constant(self, name: str) -> int:
        value = self._constant_value(name)
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        _validate_positive_byte_multiple(name, value)
        return value

    def __getattr__(self, name):
        if hasattr(self.constants, name):
            return getattr(self.constants, name)
        raise AttributeError(f"'HashConfig' object has no attribute '{name}'")

    @classmethod
    def _get_classes(cls) -> list[type]:
        """
        Get all dataclass types defined in HashConfig.
        """
        module = sys.modules[cls.__module__]
        objects = [obj for _, obj in inspect.getmembers(module, inspect.isclass) \
                if obj.__module__ == module.__name__]
        objects.remove(cls)
        return tuple(objects)

if __name__ == "__main__":
    hc = HashConfig(hash_alg="md5", length=256)
    print(hc.byteorder)
    print(hc.ws_bits)
    print(hc.bs_bits)
    print(hc.ws_bytes)
    print(hc.bs_bytes)
    print(hc.mask)
    print(HashConfig._get_classes()) #pylint: disable=protected-access
