"""
Base calculations for Hash functions.
"""

from __future__ import annotations
from typing import Sequence

import argparse

from diffusion_hash_inv.config.hash_config import HashConfig

class BaseCalc:
    """
    Base calculation class for hash functions.
    Attributes:
        word_size (int): Word size in bits.
        block_size (int): Block size in bits.
        mask (int): Mask for word size.
        byteorder (Optional[str]): Byte order ('big' or 'little').
    """

    def __init__(self, hash_config: HashConfig) -> None:
        """
        Initialize the base calculation class.  
        Parameters:
            hash_config (HashConfig): Configuration for hash algorithm.
        """
        self.word_size = hash_config.ws_bits
        self.block_size = hash_config.bs_bits
        self.byteorder = hash_config.byteorder
        self.mask = hash_config.mask

        self.sanity_check()

        # Overflow tracking
        self.overflow_boolean: bool = False
        self.total_overflow_count: int = 0
        self.loop_overflow_count: int = 0

    def sanity_check(self) -> None:
        """Perform sanity checks on the configuration."""
        assert self.word_size > 0, "word_size must be positive."
        assert self.block_size > 0, "block_size must be positive."
        assert self.block_size % self.word_size == 0, \
            "block_size must be a multiple of word_size."
        assert self.byteorder in ("big", "little"), \
            "byteorder must be either 'big' or 'little'."
        assert isinstance(self.mask, int) and self.mask >= 0, "mask must be a non-negative integer."

    def clear_overflow(self) -> None:
        """Clear overflow status."""
        self.total_overflow_count = 0
        self.loop_overflow_count = 0

    def word_to_int(self, b: Sequence[bytes | bytearray | int]) -> int:
        """
        Convert bytes to integer using the specified byteorder.

        Parameters:
            b (bytes): Input bytes in word size.
        Returns:
            int: Converted integer.
        """
        assert isinstance(b, bytes), "Input must be bytes."
        assert len(b) == self.word_size // 8, \
            f"Input bytes length must be equal to {self.word_size // 8} bytes."

        return int.from_bytes(b, self.byteorder) & self.mask

    def block_to_int(self, b: Sequence[Sequence[bytes]]) -> int:
        """
        Convert bytes to integer using the specified byteorder.
        Parameters:
            b (bytes): Input bytes in block size.
        Returns:
            int: Converted integer.
        """
        block_size_bytes = self.block_size // 8
        assert isinstance(b, bytes), "Input must be bytes."
        assert len(b) == block_size_bytes, \
            f"Input bytes length must be equal to {block_size_bytes} bytes."

        return int.from_bytes(b, self.byteorder) & self.mask

    def get_variable(self, name: str) -> int | str:
        """Retrieve the value of a variable by its name."""
        assert hasattr(self, name), f"{name} is not a valid attribute of {type(self).__name__}."
        value = getattr(self, name)
        assert isinstance(value, int | str), f"{name} is not an integer or string attribute."
        return value

    def set_variable(self, name: str, value: int | bytes) -> None:
        """Set the value of a variable by its name."""
        assert hasattr(self, name), f"{name} is not a valid attribute of {type(self).__name__}."
        assert isinstance(value, int | bytes), f"Value {value} must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert isinstance(value, int), f"Value {value} must be an integer after conversion."
        setattr(self, name, value)

    def modular_add(self, *args: int | bytes) -> int:
        """Perform modular addition on the provided integers."""
        ret = 0
        for arg in args:
            assert isinstance(arg, int | bytes), "All arguments must be integers or bytes."
            if isinstance(arg, bytes):
                arg = self.word_to_int(arg)
            assert 0 <= arg <= self.mask, "All arguments must be within word size."

            ret += arg
            if ret > self.mask:
                self.overflow_boolean = True
                self.total_overflow_count += 1
                self.loop_overflow_count += 1
            ret &= self.mask

        return ret

    def rotl(self, value: int | bytes, shift: int) -> int:
        """Perform left rotation on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= self.mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < self.word_size, "Shift must be within word size."

        return ((value << shift) | (value >> (self.word_size - shift))) & self.mask

    def rotr(self, value: int | bytes, shift: int) -> int:
        """Perform right rotation on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= self.mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < self.word_size, "Shift must be within word size."

        return ((value >> shift) | (value << (self.word_size - shift))) & self.mask

    def shr(self, value: int | bytes, shift: int) -> int:
        """Perform right shift on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= self.mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < self.word_size, "Shift must be within word size."

        return (value >> shift) & self.mask

    def shl(self, value: int | bytes, shift: int) -> int:
        """Perform left shift on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= self.mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < self.word_size, "Shift must be within word size."

        return (value << shift) & self.mask

    def modular_not(self, value: int | bytes) -> int:
        """Perform modular NOT on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= self.mask, "Value must be within word size."

        return (~value) & self.mask

    def modular_and(self, a: int | bytes, b: int | bytes) -> int:
        """Perform modular AND on the provided integers."""
        assert isinstance(a, int | bytes), "First argument must be an integer or bytes."
        assert isinstance(b, int | bytes), "Second argument must be an integer or bytes."
        if isinstance(a, bytes):
            a = self.word_to_int(a)
        if isinstance(b, bytes):
            b = self.word_to_int(b)
        assert 0 <= a <= self.mask, "First argument must be within word size."
        assert 0 <= b <= self.mask, "Second argument must be within word size."
        return (a & b) & self.mask

    def modular_or(self, a: int | bytes, b: int | bytes) -> int:
        """Perform modular OR on the provided integers."""
        assert isinstance(a, int | bytes), "First argument must be an integer or bytes."
        assert isinstance(b, int | bytes), "Second argument must be an integer or bytes."
        if isinstance(a, bytes):
            a = self.word_to_int(a)
        if isinstance(b, bytes):
            b = self.word_to_int(b)
        assert 0 <= a <= self.mask, "First argument must be within word size."
        assert 0 <= b <= self.mask, "Second argument must be within word size."

        return (a | b) & self.mask

    def modular_xor(self, a: int | bytes, b: int | bytes) -> int:
        """Perform modular XOR on the provided integers."""
        assert isinstance(a, int | bytes), "First argument must be an integer or bytes."
        assert isinstance(b, int | bytes), "Second argument must be an integer or bytes."
        if isinstance(a, bytes):
            a = self.word_to_int(a)
        if isinstance(b, bytes):
            b = self.word_to_int(b)
        assert 0 <= a <= self.mask, "First argument must be within word size."
        assert 0 <= b <= self.mask, "Second argument must be within word size."

        return (a ^ b) & self.mask


if __name__ == "__main__":
    # Example usage and simple test cases
    parser = argparse.ArgumentParser(description="Test BaseCalc operations.")
    parser.add_argument("--hash_alg", type=str, default="custom",
                        help="Hash algorithm to use (default: custom)")
    parser.add_argument("--length", type=int, default=256,
                        help="Length in bits (default: 256)")
    _args = parser.parse_args()

    _hash_config = HashConfig(
        hash_alg=_args.hash_alg,
        length=_args.length
    )

    calc = BaseCalc(hash_config=_hash_config)

    _a = calc.word_to_int(b'\x12\x34\x56\x78')
    _b = calc.word_to_int(b'\x9A\xBC\xDE\xF0')

    print(f"a: {_a:#0{10}x}")
    print(f"b: {_b:#0{10}x}")

    sum_ab = calc.modular_add(_a, _b)
    print(f"a + b: {sum_ab:#0{10}x}")

    rot_a = calc.rotl(_a, 8)
    print(f"rotl(a, 8): {rot_a:#0{10}x}")

    and_ab = calc.modular_and(_a, _b)
    print(f"a & b: {and_ab:#0{10}x}")
