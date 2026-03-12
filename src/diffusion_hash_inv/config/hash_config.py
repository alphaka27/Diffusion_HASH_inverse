"""
Configuration for hash algorithm parameters.
"""

from dataclasses import dataclass, field
import sys
from typing import Optional, Tuple, Dict
import inspect

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
        int(abs((2 ** 32) * (abs(__import__('math').sin(i + 1))))) for i in range(64)
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
        if self.word_size <= 0 or self.word_size % 8 != 0:
            raise ValueError("word_size must be a positive multiple of 8")

        if self.block_size <= 0 or self.block_size % 8 != 0:
            raise ValueError("block_size must be a positive multiple of 8")

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
        if self.word_size <= 0 or self.word_size % 8 != 0:
            raise ValueError("word_size must be a positive multiple of 8")

        if self.block_size <= 0 or self.block_size % 8 != 0:
            raise ValueError("block_size must be a positive multiple of 8")

        object.__setattr__(self, "mask", (1 << self.word_size) - 1)

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

        if hash_alg_lower == "md5":
            object.__setattr__(self, "constants", MD5Constants()) # bypass frozen

        elif hash_alg_lower == "sha256":
            object.__setattr__(self, "constants", SHA256Constants()) # bypass frozen

        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_alg}")

        print("HashConfig initialized with algorithm:", self.hash_alg)
        print("Length (bits):", self.length)

        if self.length <= 0 or self.length % 8 != 0:
            raise ValueError("length must be a positive multiple of 8")

        if self.constants is None:
            raise ValueError("Hash algorithm constants are not set.")

    @property
    def byteorder(self) -> str:
        """
        Get the byte order configuration.
        """
        if self.constants.byteorder is None:
            raise ValueError("byteorder is not set in hash algorithm constants.")

        if self.constants.byteorder not in ("little", "big"):
            raise ValueError("byteorder must be either 'little' or 'big'")

        ret = None
        if isinstance(self.constants.byteorder, str):
            ret = str(self.constants.byteorder)
        assert ret is not None, "byteorder conversion failed."

        return ret

    @property
    def ws_bits(self) -> int:
        """
        Get the word size configuration.
        """
        if self.constants.word_size is None:
            raise ValueError("word_size is not set in hash algorithm constants.")
        if self.constants.word_size < 0 or self.constants.word_size % 8 != 0:
            raise ValueError("word_size must be a positive multiple of 8")

        return self.constants.word_size

    @property
    def bs_bits(self) -> int:
        """
        Get the block size configuration.
        """
        if self.constants.block_size is None:
            raise ValueError("block_size is not set in hash algorithm constants.")
        if self.constants.block_size < 0 or self.constants.block_size % 8 != 0:
            raise ValueError("block_size must be a positive multiple of 8")

        return self.constants.block_size

    @property
    def ws_bytes(self) -> int:
        """
        Get the word size in bytes.
        """
        if self.constants.word_size is None:
            raise ValueError("word_size is not set in hash algorithm constants.")
        if self.constants.word_size < 0 or self.constants.word_size % 8 != 0:
            raise ValueError("word_size must be a positive multiple of 8")

        return self.constants.word_size // 8

    @property
    def bs_bytes(self) -> int:
        """
        Get the block size in bytes.
        """
        if self.constants.block_size is None:
            raise ValueError("block_size is not set in hash algorithm constants.")
        if self.constants.block_size < 0 or self.constants.block_size % 8 != 0:
            raise ValueError("block_size must be a positive multiple of 8")

        return self.constants.block_size // 8

    @property
    def mask(self) -> int:
        """
        Get the mask configuration.
        """
        if self.constants.mask is None:
            raise ValueError("mask is not set in hash algorithm constants.")

        return self.constants.mask

    @property
    def hierarchy(self) -> Tuple[str, str, str]:
        """
        Get the hierarchy configuration.
        """
        if self.constants.hierarchy is None:
            raise ValueError("hierarchy is not set in hash algorithm constants.")

        return self.constants.hierarchy

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(self.constants, name):
                return getattr(self.constants, name)
            raise

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
