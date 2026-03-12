"""
Defines RGB color space subcubes and provides utilities to convert byte values to RGB tuples.
Each subcube represents a partition of the RGB color space.
"""

from __future__ import annotations

from diffusion_hash_inv.utils import RGB, RGBBinning

class Byte2RGB:
    """
    A class to convert byte values(0x00 ~ 0xFF) to RGB tuples in  RGB color space.
    """

    def __init__(self, full_space_min: int = 0, full_space_max: int = 255, \
                sub_len: int = 36, split: int = 7):

        RGBBinning.config(bin_num=split, bin_width=sub_len, \
                        fr_min=full_space_min, fr_max=full_space_max)

        self.rgbbins = RGBBinning()

    def encode(self, byte: bytes) -> RGB:
        """
        Encode a byte value (0-255) to an RGB tuple.

        Args:
            byte (bytes): A single byte value to encode.

        Returns:
            RGB: The corresponding RGB tuple.
        """
        if len(byte) != 1:
            raise ValueError("Input must be a single byte.")


    def decode(self, rgb: RGB) -> bytes:
        """
        Decode an RGB tuple back to its corresponding byte value.

        Args:
            rgb (RGB): The RGB tuple to decode.

        Returns:
            bytes: The corresponding byte value.
        """




if __name__ == "__main__":
    b2rgb = Byte2RGB()
    test_byte = bytes([150])
    rgb = b2rgb.encode(test_byte)
    print(f"Byte {test_byte} encoded to RGB: {rgb}")

    decoded_byte = b2rgb.decode(rgb)
    print(f"RGB {rgb} decoded back to Byte: {decoded_byte}")
