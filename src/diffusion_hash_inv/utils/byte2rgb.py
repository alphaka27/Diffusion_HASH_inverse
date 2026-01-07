"""
Byte to RGB Color Model Conversion Module
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
#from PIL import Image

@dataclass
class RGB:
    """
    RGB Color Model Representation.
    """

    r: int
    g: int
    b: int


@dataclass
class Cordinate:
    """
    Convert RGB Color Model to Reduced Coordinate System.
    """

    sep_start:int = 0
    sep_end:int = 252
    sep_num:int = 7

    sep_point_list: tuple[int] = tuple(np.linspace(sep_start, sep_end, sep_num + 1))
    print(sep_point_list)

    def cord_valid(self):
        """
        Validates the coordinate values.
        """

    def cord_convert(self, byte: bytes):
        """
        Converts RGB values to reduced coordinate system.
        """
        assert len(byte) == 1, "Must input 1 byte for RGB encoding."






class Byte2RGB:
    """
    A class to convert byte values(0x00 ~ 0xFF) to RGB color tuples.
    """

    def __init__(self):
        self.rgb: tuple[int, int, int] = (0, 0, 0)

    def cordinate(self):
        """
        Returns the RGB color tuple.
        """

        return self.rgb

    def encoder(self):
        """
        Encodes a byte value to an RGB color tuple.
        """

    def decoder(self):
        """
        Decodes an RGB color tuple back to a byte value.
        """


if __name__ == "__main__":
    a = Cordinate()
    a.cord_convert(b"\x00")
