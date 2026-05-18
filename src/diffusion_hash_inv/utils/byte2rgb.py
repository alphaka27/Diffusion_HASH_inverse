"""
Defines RGB color space subcubes and provides utilities to convert bytes to RGB tuples.

The active encoder is bit-oriented with error correction: each RGB channel is
split into low/high halves, producing eight RGB octants.  Each byte is expanded
to a 24-bit extended Golay codeword, so one byte is stored in 24 RGB pixels.
The code can correct up to three bit/pixel classification errors per byte.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import Tuple, List

from diffusion_hash_inv.core import RGB, RGBA
from diffusion_hash_inv.core.rgb_type import Chunk1D
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.config import Byte2RGBConfig, HashConfig, MainConfig


class Byte2RGB:
    """
    Convert byte values to RGB tuples using bit-position RGB octants plus Golay ECC.
    """

    data_bits_per_byte = 8
    message_bits_per_codeword = 12
    code_bits_per_byte = 24
    golay23_generator = 0b101011100011
    golay23_parity_bits = 11
    golay23_code_bits = 23
    golay_max_correctable_bits = 3
    _golay23_syndrome_table_cache: dict[int, int] | None = None

    def __init__(self, main_config: MainConfig, \
                hash_config: HashConfig, \
                rgb_config: Byte2RGBConfig = Byte2RGBConfig(seed_flag=False, input_seed=42)):
        self.main_cfg = main_config
        self.hash_cfg = hash_config
        self.rgb_config = rgb_config
        if rgb_config.fr_min < 0 or rgb_config.fr_max > 255:
            raise ValueError("fr_min and fr_max must be in the range [0, 255].")
        if rgb_config.fr_min >= rgb_config.fr_max:
            raise ValueError("fr_min must be less than fr_max.")
        self.channel_split = (rgb_config.fr_min + rgb_config.fr_max) // 2
        self.low_chunk = Chunk1D(start=rgb_config.fr_min, end=self.channel_split)
        self.high_chunk = Chunk1D(start=self.channel_split + 1, end=rgb_config.fr_max)
        self.encoding_map = self._build_bit_chunk_encoding_map()

    @staticmethod
    def _chunk_center(chunk: Chunk1D) -> int:
        return (chunk.start + chunk.end) // 2

    def _build_bit_chunk_encoding_map(self) -> dict[int, dict[str, Chunk1D]]:
        encoding_map: dict[int, dict[str, Chunk1D]] = {}
        for octant in range(8):
            r_bit = (octant >> 2) & 1
            g_bit = (octant >> 1) & 1
            b_bit = octant & 1
            encoding_map[octant] = {
                "r_chunk": self.high_chunk if r_bit else self.low_chunk,
                "g_chunk": self.high_chunk if g_bit else self.low_chunk,
                "b_chunk": self.high_chunk if b_bit else self.low_chunk,
            }
        return encoding_map

    @staticmethod
    def _byte_to_bits(value: int) -> list[int]:
        assert 0 <= value <= 255, "Byte value must be in the range 0-255"
        return [int(bit) for bit in f"{value:08b}"]

    @classmethod
    def _int_to_bits(cls, value: int, width: int) -> list[int]:
        return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]

    @staticmethod
    def _bits_to_int(bits: list[int]) -> int:
        value = 0
        for bit in bits:
            if bit not in (0, 1):
                raise ValueError("bits must contain only 0 or 1")
            value = (value << 1) | bit
        return value

    @classmethod
    def _poly_mod(cls, value: int, generator: int) -> int:
        if generator <= 0:
            raise ValueError("generator must be positive")
        generator_degree = generator.bit_length() - 1
        while value and value.bit_length() - 1 >= generator_degree:
            shift = value.bit_length() - 1 - generator_degree
            value ^= generator << shift
        return value

    @classmethod
    def _golay23_syndrome(cls, code23: int) -> int:
        mask = (1 << cls.golay23_code_bits) - 1
        return cls._poly_mod(code23 & mask, cls.golay23_generator)

    @classmethod
    def _golay23_syndrome_table(cls) -> dict[int, int]:
        if cls._golay23_syndrome_table_cache is not None:
            return cls._golay23_syndrome_table_cache

        table: dict[int, int] = {0: 0}
        positions = range(cls.golay23_code_bits)
        for weight in range(1, cls.golay_max_correctable_bits + 1):
            for combo in combinations(positions, weight):
                pattern = 0
                for position in combo:
                    pattern |= 1 << (cls.golay23_code_bits - 1 - position)
                syndrome = cls._golay23_syndrome(pattern)
                table.setdefault(syndrome, pattern)
        cls._golay23_syndrome_table_cache = table
        return table

    @classmethod
    def _golay24_encode_byte(cls, value: int) -> list[int]:
        assert 0 <= value <= 255, "Byte value must be in the range 0-255"
        message = value << (cls.message_bits_per_codeword - cls.data_bits_per_byte)
        shifted = message << cls.golay23_parity_bits
        remainder = cls._poly_mod(shifted, cls.golay23_generator)
        code23 = shifted | remainder
        parity = code23.bit_count() & 1
        code24 = (code23 << 1) | parity
        return cls._int_to_bits(code24, cls.code_bits_per_byte)

    @classmethod
    def _golay24_decode_bits(cls, bits: list[int]) -> dict[str, object]:
        if len(bits) != cls.code_bits_per_byte:
            raise ValueError(f"expected {cls.code_bits_per_byte} Golay bits, got {len(bits)}")
        received24 = cls._bits_to_int(bits)
        received23 = received24 >> 1
        received_parity = received24 & 1
        syndrome = cls._golay23_syndrome(received23)
        table = cls._golay23_syndrome_table()
        error_pattern = table.get(syndrome)
        if error_pattern is None:
            return {
                "valid": False,
                "uncorrectable": True,
                "byte": None,
                "corrected": False,
                "corrected_bit_indices": [],
                "syndrome": syndrome,
            }

        corrected23 = received23 ^ error_pattern
        corrected_indices = [
            position
            for position in range(cls.golay23_code_bits)
            if error_pattern & (1 << (cls.golay23_code_bits - 1 - position))
        ]
        parity_mismatch = (corrected23.bit_count() ^ received_parity) & 1
        if parity_mismatch:
            corrected_indices.append(cls.golay23_code_bits)

        if len(corrected_indices) > cls.golay_max_correctable_bits:
            return {
                "valid": False,
                "uncorrectable": True,
                "byte": None,
                "corrected": False,
                "corrected_bit_indices": corrected_indices,
                "syndrome": syndrome,
            }

        message = corrected23 >> cls.golay23_parity_bits
        byte_value = message >> (cls.message_bits_per_codeword - cls.data_bits_per_byte)
        return {
            "valid": True,
            "uncorrectable": False,
            "byte": byte_value,
            "corrected": bool(corrected_indices),
            "corrected_bit_indices": corrected_indices,
            "syndrome": syndrome,
        }

    @staticmethod
    def _octant_for_bit(bit_position: int, bit_value: int) -> int:
        if bit_position < 0 or bit_position > 7:
            raise ValueError("bit_position must be in the range 0-7")
        if bit_value not in (0, 1):
            raise ValueError("bit_value must be 0 or 1")
        return bit_position if bit_value == 1 else bit_position ^ 0b111

    def _rgb_from_octant(self, octant: int) -> RGB:
        encoded = self.encoding_map[octant]
        return RGB(
            r=self._chunk_center(encoded["r_chunk"]),
            g=self._chunk_center(encoded["g_chunk"]),
            b=self._chunk_center(encoded["b_chunk"]),
        )

    def rgb_octant_decoder(self, rgb: RGB) -> int | None:
        """
        Decode one RGB value into the corresponding RGB octant index.
        """
        if not (
            self.rgb_config.fr_min <= rgb.r <= self.rgb_config.fr_max
            and self.rgb_config.fr_min <= rgb.g <= self.rgb_config.fr_max
            and self.rgb_config.fr_min <= rgb.b <= self.rgb_config.fr_max
        ):
            return None
        r_bit = 1 if rgb.r > self.channel_split else 0
        g_bit = 1 if rgb.g > self.channel_split else 0
        b_bit = 1 if rgb.b > self.channel_split else 0
        return (r_bit << 2) | (g_bit << 1) | b_bit

    def rgb_bit_chunk_decoder(self, rgb: RGB) -> int | None:
        """Compatibility alias for callers that expect an integer RGB space id."""
        return self.rgb_octant_decoder(rgb)

    @staticmethod
    def _octants_to_raw_bits(octants: list[int]) -> list[int]:
        return [
            1 if octant == (index % Byte2RGB.data_bits_per_byte) else 0
            for index, octant in enumerate(octants)
        ]

    @classmethod
    def decode_octants(cls, octants: list[int]) -> tuple[list[int], list[dict[str, object]]]:
        """
        Decode RGB octants into bytes and per-byte Golay correction reports.
        """
        values: list[int] = []
        reports: list[dict[str, object]] = []
        usable = len(octants) - (len(octants) % cls.code_bits_per_byte)
        for offset in range(0, usable, cls.code_bits_per_byte):
            group = octants[offset:offset + cls.code_bits_per_byte]
            bits = cls._octants_to_raw_bits(group)
            report = cls._golay24_decode_bits(bits)
            report["index"] = offset // cls.code_bits_per_byte
            report["raw_bits"] = "".join(str(bit) for bit in bits)
            reports.append(report)
            byte_value = report.get("byte")
            if report.get("valid") and isinstance(byte_value, int):
                values.append(byte_value)
        return values, reports

    @classmethod
    def octants_to_byte_values(cls, octants: list[int]) -> list[int]:
        """
        Convert decoded octant sequence into byte integers.

        One byte requires one 24-bit extended Golay codeword.  Incomplete
        trailing octants are ignored.
        """
        values, _reports = cls.decode_octants(octants)
        return values

    def bit_chunks_to_byte_values(self, chunks: list[int]) -> list[int]:
        """Compatibility alias for decoded RGB octant sequences."""
        return self.octants_to_byte_values(chunks)

    def _rgb_encoding(self, hexstring: str, byteorder: str) -> RGB | Tuple[RGB, ...]:
        """
        Encode bytes/hex string into Golay-protected bit-position RGB tuples.

        Args:
            hexstring (str): A hexadecimal string representing the byte value to encode.

        Returns:
            RGB | Tuple[RGB, ...]: The corresponding RGB tuple(s).
        """
        bytes_value = Logs.str_to_bytes(hexstring) if isinstance(hexstring, str) else hexstring
        int_value = Logs.bytes_to_int(bytes_value, byteorder=byteorder)
        encode = []
        for integer in int_value:
            for code_position, bit_value in enumerate(self._golay24_encode_byte(integer)):
                octant = self._octant_for_bit(code_position % self.data_bits_per_byte, bit_value)
                encode.append(self._rgb_from_octant(octant))

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
        rgb_values = (rgb,) if isinstance(rgb, RGB) else rgb
        if not isinstance(rgb_values, tuple) or not all(isinstance(item, RGB) for item in rgb_values):
            raise TypeError("Input must be an RGB instance or a tuple of RGB instances.")

        decode_octants: List[int] = []
        for _rgb in rgb_values:
            octant = self.rgb_octant_decoder(_rgb)
            if octant is not None:
                decode_octants.append(octant)

        byte_values = self.octants_to_byte_values(decode_octants)
        decode_bytes = Logs.iter_to_bytes(byte_values, byteorder=self.hash_cfg.byteorder)

        if self.main_cfg.verbose_flag:
            print(f"Decoded RGB: {rgb} to byte value: {decode_bytes}")

        return decode_bytes



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hash Generation and Image Creation Script")
    parser.add_argument('--mode', type=str, default="verify",
                        help='Mode of operation (default: verify)')
    parser.add_argument('-i', '--input', type=str, default=argparse.SUPPRESS,
                        help='Input value for testing (default: 0x89abcdef)')
    parser.add_argument('--hash_alg', type=str, default='md5',
                        help='Hash algorithm to use (default: md5)')
    _args = parser.parse_args()
    _main_cfg = MainConfig(
            message_flag=True,
            verbose_flag=True,
            clean_flag=False,
            debug_flag=False,
            make_xlsx_flag=False,
            seed_flag=False,
        )
    _hash_cfg = HashConfig(hash_alg=_args.hash_alg, length=1024)
    b2rgb = Byte2RGB(_main_cfg, _hash_cfg)
    print("----- Byte to RGB Encoding Test -----")
    TEST_HEX = "0x89abcdef"
    test_byte = Logs.str_to_bytes(TEST_HEX)
    _rgb = b2rgb.rgb_encoder(test_byte)

    print()

    print("----- RGB to Byte Decoding Test -----")
    DECODE = b2rgb.rgb_decoder(_rgb)

    assert DECODE == test_byte, "Decoded byte does not match the original byte."
