"""
Defines RGB color space subcubes and provides utilities to convert byte values to RGB tuples.
Each subcube represents a partition of the RGB color space.
"""

from __future__ import annotations

from secrets import randbelow
from typing import Tuple, List

from diffusion_hash_inv.core import RGB, RGBBinning
from diffusion_hash_inv.core import Logs
from diffusion_hash_inv.config import Byte2RGBConfig, HashConfig, MainConfig

class Byte2RGB:
    """
    A class to convert byte values(0x00 ~ 0xFF) to RGB tuples in  RGB color space.
    """

    def __init__(self, rgb_config: Byte2RGBConfig = Byte2RGBConfig(), \
                hash_config: HashConfig = HashConfig("md5", 256)):
        self.byteorder = hash_config.byteorder
        binning = RGBBinning()
        binning.config(
            bin_num=rgb_config.bin_num,
            bin_width=rgb_config.bin_width,
            fr_min=rgb_config.fr_min,
            fr_max=rgb_config.fr_max,
        )
        rgbbins = binning()
        self.encoding_map = {}
        for _i, _bin in enumerate(rgbbins):
            self.encoding_map[_i] = {
                "r_chunk": _bin.r_chunk,
                "g_chunk": _bin.g_chunk,
                "b_chunk": _bin.b_chunk,
            }

    def encode(self, hexstring: str | bytes) -> RGB | Tuple[RGB, ...]:
        """
        Encode a byte value (0-255) to an RGB tuple.

        Args:
            hexstring (str): A hexadecimal string representing the byte value to encode.

        Returns:
            RGB: The corresponding RGB tuple.
        """
        bytes_value = Logs.str_to_bytes(hexstring) if isinstance(hexstring, str) else hexstring
        int_value = Logs.bytes_to_int(bytes_value, byteorder=self.byteorder)
        encode = []
        for integer in int_value:
            assert 0 <= integer <= 255, "Byte value must be in the range 0-255"
            _temp_encode = self.encoding_map.get(integer)
            _temp_r = _temp_encode["r_chunk"].as_half_open
            _temp_g = _temp_encode["g_chunk"].as_half_open
            _temp_b = _temp_encode["b_chunk"].as_half_open
            _r = randbelow(_temp_r.end - _temp_r.start) + _temp_r.start
            _g = randbelow(_temp_g.end - _temp_g.start) + _temp_g.start
            _b = randbelow(_temp_b.end - _temp_b.start) + _temp_b.start
            encode.append(RGB(r=_r, g=_g, b=_b))

        if MainConfig.verbose_flag:
            print(f"Encoded byte value: {bytes_value} to RGB: {encode}")

        if len(encode) == 1:
            return encode[0]
        return tuple(encode)

    def decode(self, rgb: RGB | Tuple[RGB, ...]) -> bytes:
        """
        Decode an RGB tuple back to its corresponding byte value.

        Args:
            rgb (RGB): The RGB tuple to decode.

        Returns:
            bytes: The corresponding byte value.
        """
        decode: List[int] = []
        assert isinstance(rgb, Tuple), "Input must be an RGB instance or a tuple of RGB instances."

        for _rgb in rgb:
            _r, _g, _b = _rgb.r, _rgb.g, _rgb.b
            for key, val in self.encoding_map.items():
                r_chunk = val["r_chunk"].as_inclusive
                g_chunk = val["g_chunk"].as_inclusive
                b_chunk = val["b_chunk"].as_inclusive
                if (r_chunk.start <= _r <= r_chunk.end) and \
                    (g_chunk.start <= _g <= g_chunk.end) and \
                    (b_chunk.start <= _b <= b_chunk.end):
                    decode.append(key)
                    continue

        decode_bytes = Logs.iter_to_bytes(decode, byteorder=self.byteorder)

        if MainConfig.verbose_flag:
            print(f"Decoded RGB: {rgb} to byte value: {decode_bytes}")

        return decode_bytes



if __name__ == "__main__":
    b2rgb = Byte2RGB()

    print("----- Byte to RGB Encoding Test -----")
    test_byte = Logs.str_to_bytes("0x6e4c5a2e")
    _rgb = b2rgb.encode(test_byte)

    print()

    print("----- RGB to Byte Decoding Test -----")
    DECODE = b2rgb.decode(_rgb)

    assert DECODE == test_byte, "Decoded byte does not match the original byte."
