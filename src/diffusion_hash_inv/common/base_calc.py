"""
Base calculations for Hash functions.
"""

from __future__ import annotations
from typing import Optional, ClassVar, Sequence

class BaseCalc:
    """
    Base calculation class for hash functions.
    Attributes:
        word_size (int): Word size in bits.
        block_size (int): Block size in bits.
        mask (int): Mask for word size.
        byteorder (Optional[str]): Byte order ('big' or 'little').
    """
    word_size: ClassVar[int]
    block_size: ClassVar[int]
    ws_byte: ClassVar[int]
    bs_byte: ClassVar[int]
    mask: ClassVar[int]
    byteorder: ClassVar[Optional[str]] = None

    def __init__(self, word_size: int, block_size: int, byteorder: Optional[str] = None) -> None:
        """
        Initialize the base calculation class.  
        Parameters:
            word_size (int): Word size in bits.
            block_size (int): Block size in bits.
            byteorder (Optional[str]): Byte order ('big' or 'little'). Default is None.
        """
        type(self).word_size = word_size
        type(self).block_size = block_size
        type(self).ws_byte = word_size // 8
        type(self).bs_byte = block_size // 8
        type(self).mask = (1 << word_size) - 1
        type(self).byteorder = byteorder
        self.overflow_boolean: bool = False
        self.total_overflow_count: int = 0
        self.loop_overflow_count: int = 0
        assert type(self).byteorder in ('big', 'little'), "Byteorder must be 'big' or 'little'."

    def clear_overflow(self) -> None:
        """Clear overflow status."""
        self.total_overflow_count = 0
        self.loop_overflow_count = 0

    @staticmethod
    def byte_to_int(b: bytes, byteorder: Optional[str] = None) -> int:
        """
        Convert bytes to integer using the specified byteorder.
        Parameters:
            b (bytes): Input byte.
        Returns:
            int: Converted integer.
        """
        assert isinstance(b, bytes), "Input must be bytes."
        assert byteorder in ('big', 'little'), "Byteorder must be 'big' or 'little'."

        return int.from_bytes(b, byteorder)

    def word_to_int(self, b: Sequence[bytes | bytearray | int]) -> int:
        """
        Convert bytes to integer using the specified byteorder.

        Parameters:
            b (bytes): Input bytes in word size.
        Returns:
            int: Converted integer.
        """
        assert isinstance(b, bytes), "Input must be bytes."
        assert len(b) == type(self).ws_byte, \
            f"Input bytes length must be equal to {type(self).ws_byte} bytes."

        return int.from_bytes(b, type(self).byteorder) & type(self).mask

    def block_to_int(self, b: Sequence[Sequence[bytes]]) -> int:
        """
        Convert bytes to integer using the specified byteorder.
        Parameters:
            b (bytes): Input bytes in block size.
        Returns:
            int: Converted integer.
        """
        assert isinstance(b, bytes), "Input must be bytes."
        assert len(b) == type(self).bs_byte, \
            f"Input bytes length must be equal to {type(self).bs_byte} bytes."

        return int.from_bytes(b, type(self).byteorder) & ((1 << type(self).block_size) - 1)

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
            assert 0 <= arg <= type(self).mask, "All arguments must be within word size."

            ret += arg
            if ret > type(self).mask:
                self.overflow_boolean = True
                self.total_overflow_count += 1
                self.loop_overflow_count += 1
            ret &= type(self).mask

        return ret

    def rotl(self, value: int | bytes, shift: int) -> int:
        """Perform left rotation on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= type(self).mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < type(self).word_size, "Shift must be within word size."

        ws = type(self).word_size

        return ((value << shift) | (value >> (ws - shift))) & type(self).mask

    def rotr(self, value: int | bytes, shift: int) -> int:
        """Perform right rotation on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= type(self).mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < type(self).word_size, "Shift must be within word size."

        ws = type(self).word_size

        return ((value >> shift) | (value << (ws - shift))) & type(self).mask

    def shr(self, value: int | bytes, shift: int) -> int:
        """Perform right shift on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= type(self).mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < type(self).word_size, "Shift must be within word size."

        return (value >> shift) & type(self).mask

    def shl(self, value: int | bytes, shift: int) -> int:
        """Perform left shift on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= type(self).mask, "Value must be within word size."
        assert isinstance(shift, int), "Shift must be an integer."
        assert 0 <= shift < type(self).word_size, "Shift must be within word size."

        return (value << shift) & type(self).mask

    def modular_not(self, value: int | bytes) -> int:
        """Perform modular NOT on the provided integer."""
        assert isinstance(value, int | bytes), "Value must be an integer or bytes."
        if isinstance(value, bytes):
            value = self.word_to_int(value)
        assert 0 <= value <= type(self).mask, "Value must be within word size."

        return (~value) & type(self).mask

    def modular_and(self, a: int | bytes, b: int | bytes) -> int:
        """Perform modular AND on the provided integers."""
        assert isinstance(a, int | bytes), "First argument must be an integer or bytes."
        assert isinstance(b, int | bytes), "Second argument must be an integer or bytes."
        if isinstance(a, bytes):
            a = self.word_to_int(a)
        if isinstance(b, bytes):
            b = self.word_to_int(b)
        assert 0 <= a <= type(self).mask, "First argument must be within word size."
        assert 0 <= b <= type(self).mask, "Second argument must be within word size."
        return (a & b) & type(self).mask

    def modular_or(self, a: int | bytes, b: int | bytes) -> int:
        """Perform modular OR on the provided integers."""
        assert isinstance(a, int | bytes), "First argument must be an integer or bytes."
        assert isinstance(b, int | bytes), "Second argument must be an integer or bytes."
        if isinstance(a, bytes):
            a = self.word_to_int(a)
        if isinstance(b, bytes):
            b = self.word_to_int(b)
        assert 0 <= a <= type(self).mask, "First argument must be within word size."
        assert 0 <= b <= type(self).mask, "Second argument must be within word size."

        return (a | b) & type(self).mask

    def modular_xor(self, a: int | bytes, b: int | bytes) -> int:
        """Perform modular XOR on the provided integers."""
        assert isinstance(a, int | bytes), "First argument must be an integer or bytes."
        assert isinstance(b, int | bytes), "Second argument must be an integer or bytes."
        if isinstance(a, bytes):
            a = self.word_to_int(a)
        if isinstance(b, bytes):
            b = self.word_to_int(b)
        assert 0 <= a <= type(self).mask, "First argument must be within word size."
        assert 0 <= b <= type(self).mask, "Second argument must be within word size."

        return (a ^ b) & type(self).mask
