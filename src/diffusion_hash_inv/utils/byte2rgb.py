"""
Defines RGB color space subcubes and provides utilities to convert byte values to RGB tuples.
Each subcube represents a partition of the RGB color space.
"""

from __future__ import annotations

from secrets import randbelow
from typing import Tuple, List

from diffusion_hash_inv.core import RGB, RGBA, RGBBinning
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.config import Byte2RGBConfig, HashConfig, MainConfig


class Byte2RGB:
    """
    A class to convert byte values(0x00 ~ 0xFF) to RGB tuples in  RGB color space.
    """

    def __init__(self, main_config: MainConfig, \
                hash_config: HashConfig, \
                rgb_config: Byte2RGBConfig = Byte2RGBConfig()):
        self.main_cfg = main_config
        self.hash_cfg = hash_config
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

    def _rgb_encoding(self, hexstring: str, byteorder: str) -> RGB | Tuple[RGB, ...]:
        """
        Encoding hexstring to RGB tuple

        Args:
            hexstring (str): A hexadecimal string representing the byte value to encode.

        Returns:
            RGB: The corresponding RGB tuple.
        """
        bytes_value = Logs.str_to_bytes(hexstring) if isinstance(hexstring, str) else hexstring
        int_value = Logs.bytes_to_int(bytes_value, byteorder=byteorder)
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

        if self.main_cfg.verbose_flag:
            print(f"Encoded byte value: {bytes_value} or {hexstring} to RGB: \n{encode}")


        if len(encode) == 1:
            return encode[0]
        return tuple(encode)

    # TODO: Implement RGBA encoding
    def _rgba_encoding(self, hexstring: str | bytes, byteorder: str):
        """
        Encoding hexstring to RGBA tuple
        """
        alpha_max = 255

        raise NotImplementedError("RGBA encoding is not yet implemented.")

    def rgb_encoder(self, hexstring: str | bytes, encoding: str = "RGB") \
            -> RGB | RGBA | Tuple[RGB, ...] | Tuple[RGBA, ...]:
        """
        Encoding hexstring to RGB tuple or RGBA tuple depends on encoding

        Args:
            hexstring (str | bytes): A hexadecimal string or bytes representing the byte value.
            byteorder (str): The byte order to use for encoding ("big" or "little").
            encoding (str): The encoding type, either "RGB" or "RGBA".

        Returns:
            RGB | RGBA | Tuple[RGB, ...] | Tuple[RGBA, ...]: The corresponding RGB or RGBA tuple(s).
        """
        if encoding == "RGB":
            ret = self._rgb_encoding(hexstring, self.hash_cfg.byteorder)
        elif encoding == "RGBA":
            ret = self._rgba_encoding(hexstring, self.hash_cfg.byteorder) # pylint: disable=assignment-from-no-return
        else:
            raise ValueError("Unsupported encoding type. Use 'RGB' or 'RGBA'.")

        return ret


    def rgb_decoder(self, rgb: RGB | Tuple[RGB, ...]) -> bytes:
        """
        Decode an RGB tuple back to its corresponding byte value.

        Args:
            rgb (RGB | Tuple[RGB, ...]): The RGB tuple or tuples to decode.

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

        decode_bytes = Logs.iter_to_bytes(decode, byteorder=self.hash_cfg.byteorder)

        if self.main_cfg.verbose_flag:
            print(f"Decoded RGB: {rgb} to byte value: {decode_bytes}")

        return decode_bytes



if __name__ == "__main__":
    _main_cfg = MainConfig(
        message_flag=True,
        verbose_flag=True,
        clean_flag=False,
        debug_flag=False,
        make_xlsx_flag=False,
    )
    _hash_cfg = HashConfig(hash_alg="md5", length=1024)


    b2rgb = Byte2RGB(_main_cfg, _hash_cfg)

    print("----- Byte to RGB Encoding Test -----")
    TEST_HEX = "0x306a75277e7e2a7c6d7a6451283f3c7667456342672b37723c50395b375c702d"
    test_byte = Logs.str_to_bytes(TEST_HEX)
    _rgb = b2rgb.rgb_encoder(test_byte)

    print()

    print("----- RGB to Byte Decoding Test -----")
    DECODE = b2rgb.rgb_decoder(_rgb)

    assert DECODE == test_byte, "Decoded byte does not match the original byte."
