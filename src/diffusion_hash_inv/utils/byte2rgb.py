"""
Defines RGB color space subcubes and provides utilities to convert bytes to RGB tuples.

The default encoder stores each byte as a 48-bit error-correcting codeword
packed into two RGB pixels.  Legacy and cube-id encoders are kept for direct
one-pixel-per-byte mappings.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import Tuple, List

from diffusion_hash_inv.core import RGB, RGBA, RGBBinning
from diffusion_hash_inv.core.rgb_type import Chunk1D, RGBBin
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.config import Byte2RGBConfig, HashConfig, MainConfig
from diffusion_hash_inv.utils.ecc48 import DecodeResult, get_codec, SUPPORTED_METHODS


class Byte2RGB:
    """
    Convert byte values to RGB tuples using the configured RGB encoding method.
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
        if rgb_config.encoding not in ("golay24", "legacy-bin", "cube-id", *SUPPORTED_METHODS):
            raise ValueError(
                f"encoding must be one of 'golay24', 'legacy-bin', 'cube-id', "
                f"{', '.join(repr(m) for m in SUPPORTED_METHODS)}, "
                f"got '{rgb_config.encoding}'")
        self.channel_split = (rgb_config.fr_min + rgb_config.fr_max) // 2
        self.low_chunk = Chunk1D(start=rgb_config.fr_min, end=self.channel_split)
        self.high_chunk = Chunk1D(start=self.channel_split + 1, end=rgb_config.fr_max)
        self.encoding_map = self._build_bit_chunk_encoding_map()
        self._legacy_bin_map: list[RGBBin] | None = None

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

    # ------------------------------------------------------------------
    # Legacy bin encoding (encoding="legacy-bin"): 1 pixel per byte
    # ------------------------------------------------------------------

    @property
    def pixels_per_byte(self) -> int:
        """Pixels required to encode one byte.

        - ``legacy-bin`` : 1
        - ``cube-id``    : 1
        - ``golay24``    : 24
        - ``linear48``, ``golay24-dual``, ``rs48``, ``bch48`` : 2  (48-bit codecs)
        """
        enc = self.rgb_config.encoding
        if enc in ("legacy-bin", "cube-id"):
            return 1
        if enc in SUPPORTED_METHODS:
            return 2
        return self.code_bits_per_byte  # "golay24"

    @property
    def legacy_bin_map(self) -> list[RGBBin]:
        """Lazily-built ordered list of 256 RGBBin objects for legacy-bin encoding."""
        if self._legacy_bin_map is None:
            self._legacy_bin_map = self._build_legacy_bin_map()
        return self._legacy_bin_map

    def _build_legacy_bin_map(self) -> list[RGBBin]:
        bins = RGBBinning().quantization()
        if len(bins) != 256:
            raise ValueError(
                f"Legacy bin map requires exactly 256 included bins, got {len(bins)}. "
                "Check RGBBinning bin_num/bin_width settings."
            )
        return bins

    def _legacy_encode_byte(self, value: int) -> RGB:
        """Map a byte value (0–255) to the centre RGB of its corresponding bin."""
        assert 0 <= value <= 255, "Byte value must be in the range 0-255"
        b = self.legacy_bin_map[value]
        return RGB(
            r=(b.r_chunk.start + b.r_chunk.end) // 2,
            g=(b.g_chunk.start + b.g_chunk.end) // 2,
            b=(b.b_chunk.start + b.b_chunk.end) // 2,
        )

    def _legacy_decode_rgb(self, rgb: RGB) -> int | None:
        """Return the byte value (0–255) for an RGB pixel, or None if outside all bins."""
        for byte_value, b in enumerate(self.legacy_bin_map):
            if (b.r_chunk.is_in_chunk(rgb.r) and
                    b.g_chunk.is_in_chunk(rgb.g) and
                    b.b_chunk.is_in_chunk(rgb.b)):
                return byte_value
        return None

    # ------------------------------------------------------------------
    # Encoding Method 2: RGB cube-id mapping (encoding="cube-id")
    # ------------------------------------------------------------------

    @staticmethod
    def cube_id_to_rgb(cube_id: int) -> RGB:
        """
        Convert an 8-bit cube id to the center RGB value of its quantization cube.

        Inverse of the formula from ``Encoding Method.md``:

            cube_id = floor(B / 64) * 64 + floor(G / 32) * 8 + floor(R / 32)
        """
        assert 0 <= cube_id <= 255, "Cube ID must be in the range 0-255"
        r_idx = cube_id & 0b111
        g_idx = (cube_id >> 3) & 0b111
        b_idx = (cube_id >> 6) & 0b11
        return RGB(
            r=r_idx * 32 + 16,
            g=g_idx * 32 + 16,
            b=b_idx * 64 + 32,
        )

    @staticmethod
    def rgb_to_cube_id(rgb: RGB) -> int | None:
        """
        Convert an RGB value to the cube id defined in ``Encoding Method.md``.
        """
        if not (0 <= rgb.r <= 255 and 0 <= rgb.g <= 255 and 0 <= rgb.b <= 255):
            return None
        return (rgb.b // 64) * 64 + (rgb.g // 32) * 8 + (rgb.r // 32)

    def _cube_id_encode_byte(self, value: int) -> RGB:
        """Map a byte value (0–255) to the center RGB of its Method 2 cube."""
        return self.cube_id_to_rgb(value)

    def _cube_id_decode_rgb(self, rgb: RGB) -> int | None:
        """Return the Method 2 cube id for one RGB pixel."""
        return self.rgb_to_cube_id(rgb)

    # ------------------------------------------------------------------
    # 48-bit RGB pair helpers (encoding = linear48 | golay24-dual | rs48 | bch48)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_codeword_to_rgb_pair(codeword: bytes) -> tuple[RGB, RGB]:
        """
        Pack a 6-byte (48-bit) ECC codeword into two RGB pixels.

        Layout (from ``Encoding Method.md``):
            RGB_1 = (C0, C1, C2)
            RGB_2 = (C3, C4, C5)
        """
        if len(codeword) != 6:
            raise ValueError(f"Expected 6 bytes for 48-bit codeword, got {len(codeword)}")
        return (
            RGB(r=codeword[0], g=codeword[1], b=codeword[2]),
            RGB(r=codeword[3], g=codeword[4], b=codeword[5]),
        )

    @staticmethod
    def _unpack_rgb_pair_to_codeword(rgb1: RGB, rgb2: RGB) -> bytes:
        """Reverse of :meth:`_pack_codeword_to_rgb_pair`."""
        return bytes([rgb1.r, rgb1.g, rgb1.b, rgb2.r, rgb2.g, rgb2.b])

    @staticmethod
    def rgb_pair_to_2x2_patch(rgb1: RGB, rgb2: RGB) -> tuple[tuple[RGB, RGB], tuple[RGB, RGB]]:
        """
        Return the 2×2 image layout defined in ``Encoding Method.md``.

        Layout:

            RGB1 RGB2
            RGB2 RGB1
        """
        return ((rgb1, rgb2), (rgb2, rgb1))

    @staticmethod
    def rgb_pair_from_2x2_patch(
        patch: tuple[tuple[RGB, RGB], tuple[RGB, RGB]]
    ) -> tuple[RGB, RGB]:
        """
        Recover ``(RGB1, RGB2)`` from a Method 1 2×2 anti-diagonal patch.
        """
        if (
            len(patch) != 2
            or any(len(row) != 2 for row in patch)
            or any(not isinstance(pixel, RGB) for row in patch for pixel in row)
        ):
            raise ValueError("patch must be a 2x2 tuple of RGB values")
        rgb1, rgb2 = patch[0]
        if patch[1][0] != rgb2 or patch[1][1] != rgb1:
            raise ValueError("patch does not match the RGB1/RGB2 anti-diagonal layout")
        return rgb1, rgb2

    def decode_payload_with_confidence(self, rgb1: RGB, rgb2: RGB) -> DecodeResult:
        """
        Decode one RGB pixel pair into a payload byte + reliability metadata.

        Only valid for 48-bit encoding modes (``'linear48'``, ``'golay24-dual'``,
        ``'rs48'``, ``'bch48'``).  Raises ``RuntimeError`` for one-pixel and
        legacy ``'golay24'`` encodings.

        Parameters
        ----------
        rgb1, rgb2 : RGB
            The two consecutive pixels that store one encoded byte.

        Returns
        -------
        DecodeResult
            Contains ``payload`` (int 0–255 or None), ``confidence`` (float),
            ``errors_corrected``, ``method``, ``uncorrectable``, and ``detail``.
        """
        enc = self.rgb_config.encoding
        if enc not in SUPPORTED_METHODS:
            raise RuntimeError(
                f"decode_payload_with_confidence() requires a 48-bit encoding, "
                f"got '{enc}'.  Use one of: {list(SUPPORTED_METHODS)}")
        codec = get_codec(enc)
        codeword = self._unpack_rgb_pair_to_codeword(rgb1, rgb2)
        return codec.decode(codeword)

    def decode_rgb_pixel(self, rgb: RGB) -> int | None:
        """
        Decode one RGB pixel into its encoded integer value.

        - ``legacy-bin``: returns the byte value (0–255).
        - ``cube-id``: returns the Method 2 cube id (0–255).
        - ``golay24``: returns the RGB octant index (0–7).
        """
        if self.rgb_config.encoding == "legacy-bin":
            return self._legacy_decode_rgb(rgb)
        if self.rgb_config.encoding == "cube-id":
            return self._cube_id_decode_rgb(rgb)
        return self.rgb_octant_decoder(rgb)

    # ------------------------------------------------------------------
    # Public encoder / decoder (dispatch by encoding mode)
    # ------------------------------------------------------------------

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
        if self.rgb_config.encoding == "legacy-bin":
            if encoding != "RGB":
                raise ValueError("Legacy-bin encoding only supports RGB mode.")
            bytes_value = Logs.str_to_bytes(hexstring) if isinstance(hexstring, str) else hexstring
            int_values = Logs.bytes_to_int(bytes_value, byteorder=self.hash_cfg.byteorder)
            encoded = [self._legacy_encode_byte(v) for v in int_values]
            return encoded[0] if len(encoded) == 1 else tuple(encoded)

        if self.rgb_config.encoding == "cube-id":
            if encoding != "RGB":
                raise ValueError("Cube-id encoding only supports RGB mode.")
            bytes_value = Logs.str_to_bytes(hexstring) if isinstance(hexstring, str) else hexstring
            int_values = Logs.bytes_to_int(bytes_value, byteorder=self.hash_cfg.byteorder)
            encoded = [self._cube_id_encode_byte(v) for v in int_values]
            return encoded[0] if len(encoded) == 1 else tuple(encoded)

        if self.rgb_config.encoding in SUPPORTED_METHODS:
            if encoding != "RGB":
                raise ValueError("48-bit ECC encodings only support RGB mode.")
            bytes_value = Logs.str_to_bytes(hexstring) if isinstance(hexstring, str) else hexstring
            int_values = Logs.bytes_to_int(bytes_value, byteorder=self.hash_cfg.byteorder)
            codec = get_codec(self.rgb_config.encoding)
            rgb_pairs: list[RGB] = []
            for v in int_values:
                codeword = codec.encode(v)
                rgb1, rgb2 = self._pack_codeword_to_rgb_pair(codeword)
                rgb_pairs.extend([rgb1, rgb2])
            return rgb_pairs[0] if len(rgb_pairs) == 1 else tuple(rgb_pairs)

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

        if self.rgb_config.encoding == "legacy-bin":
            byte_values = [
                v for _rgb in rgb_values
                if (v := self._legacy_decode_rgb(_rgb)) is not None
            ]
            decode_bytes = Logs.iter_to_bytes(byte_values, byteorder=self.hash_cfg.byteorder)
            if self.main_cfg.verbose_flag:
                print(f"Decoded RGB: {rgb} to byte value: {decode_bytes}")
            return decode_bytes

        if self.rgb_config.encoding == "cube-id":
            byte_values = [
                v for _rgb in rgb_values
                if (v := self._cube_id_decode_rgb(_rgb)) is not None
            ]
            decode_bytes = Logs.iter_to_bytes(byte_values, byteorder=self.hash_cfg.byteorder)
            if self.main_cfg.verbose_flag:
                print(f"Decoded RGB: {rgb} to byte value: {decode_bytes}")
            return decode_bytes

        if self.rgb_config.encoding in SUPPORTED_METHODS:
            codec = get_codec(self.rgb_config.encoding)
            byte_values = []
            pairs = list(rgb_values)
            for i in range(0, len(pairs) - 1, 2):
                result = self.decode_payload_with_confidence(pairs[i], pairs[i + 1])
                if result.valid and result.payload is not None:
                    byte_values.append(result.payload)
            decode_bytes = Logs.iter_to_bytes(byte_values, byteorder=self.hash_cfg.byteorder)
            if self.main_cfg.verbose_flag:
                print(f"Decoded RGB: {rgb} to byte value: {decode_bytes}")
            return decode_bytes

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
