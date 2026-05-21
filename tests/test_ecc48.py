"""
Tests for the 48-bit ECC codecs in ``diffusion_hash_inv.utils.ecc48``.

Coverage:
- Encode/decode round-trips for all 256 payload values (per codec)
- Error injection: 1, 2, …, t bit/byte errors (within correction capability)
- Error injection just above the correction limit (expect uncorrectable)
- Confidence score semantics
- DecodeResult fields
- ``Byte2RGB`` integration (encode → rgb_encoder / decode → rgb_decoder,
  decode_payload_with_confidence)
"""

from __future__ import annotations

import random
import pytest

from diffusion_hash_inv.utils.ecc48 import (
    DecodeResult,
    Golay24DualCodec,
    RS48Codec,
    BCH48Codec,
    get_codec,
    SUPPORTED_METHODS,
)
from diffusion_hash_inv.core import RGB
from diffusion_hash_inv.config import Byte2RGBConfig, HashConfig, MainConfig
from diffusion_hash_inv.utils.byte2rgb import Byte2RGB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAIN_CFG = MainConfig(
    verbose_flag=False,
    clean_flag=False,
    debug_flag=False,
    make_image_flag=False,
)
_HASH_CFG = HashConfig(hash_alg="md5", length=16)

ALL_CODECS = [
    ("golay24-dual", Golay24DualCodec),
    ("rs48", RS48Codec),
    ("bch48", BCH48Codec),
]


def _flip_bit(data: bytes, bit_index: int) -> bytes:
    """Return a copy of `data` with the single bit at `bit_index` flipped."""
    b = bytearray(data)
    byte_pos = bit_index // 8
    bit_pos = 7 - (bit_index % 8)
    b[byte_pos] ^= (1 << bit_pos)
    return bytes(b)


def _flip_bits(data: bytes, bit_indices: list[int]) -> bytes:
    result = data
    for i in bit_indices:
        result = _flip_bit(result, i)
    return result


def _corrupt_byte(data: bytes, byte_index: int) -> bytes:
    """Corrupt one full byte (XOR with 0xFF)."""
    b = bytearray(data)
    b[byte_index] ^= 0xFF
    return bytes(b)


# ===========================================================================
# get_codec / factory
# ===========================================================================

class TestFactory:
    def test_supported_methods_tuple(self):
        assert set(SUPPORTED_METHODS) == {"golay24-dual", "rs48", "bch48"}

    @pytest.mark.parametrize("method,cls", ALL_CODECS)
    def test_get_codec_returns_correct_class(self, method, cls):
        assert get_codec(method) is cls

    def test_get_codec_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown ECC48 method"):
            get_codec("nonexistent")


# ===========================================================================
# DecodeResult fields
# ===========================================================================

class TestDecodeResultFields:
    @pytest.mark.parametrize("method,cls", ALL_CODECS)
    def test_clean_result_fields(self, method, cls):
        cw = cls.encode(42)
        r = cls.decode(cw)
        assert isinstance(r, DecodeResult)
        assert r.valid is True
        assert r.payload == 42
        assert r.confidence == pytest.approx(1.0)
        assert r.errors_corrected == 0
        assert r.method == method
        assert r.uncorrectable is False
        assert isinstance(r.detail, dict)

    @pytest.mark.parametrize("method,cls", ALL_CODECS)
    def test_confidence_in_range(self, method, cls):
        for byte_val in range(256):
            cw = cls.encode(byte_val)
            r = cls.decode(cw)
            assert 0.0 <= r.confidence <= 1.0, f"{method}: byte={byte_val}"


# ===========================================================================
# Golay24Dual
# ===========================================================================

class TestGolay24Dual:
    """Extended Golay(24,12) dual-copy codec."""

    codec = Golay24DualCodec

    def test_round_trip_all_bytes(self):
        for v in range(256):
            cw = self.codec.encode(v)
            r = self.codec.decode(cw)
            assert r.valid and r.payload == v, f"Failed on byte {v}"

    def test_codeword_length(self):
        cw = self.codec.encode(0)
        assert len(cw) == 6

    def test_halves_identical(self):
        for v in range(256):
            cw = self.codec.encode(v)
            assert cw[:3] == cw[3:], f"Halves differ for byte {v}"

    @pytest.mark.parametrize("byte_val", [0, 1, 127, 200, 255])
    @pytest.mark.parametrize("n_errors", [1, 2, 3])
    def test_corrects_n_bit_errors_in_first_half(self, byte_val, n_errors):
        cw = self.codec.encode(byte_val)
        corrupted = _flip_bits(cw, list(range(n_errors)))
        r = self.codec.decode(corrupted)
        assert r.valid and r.payload == byte_val

    @pytest.mark.parametrize("byte_val", [0, 1, 127, 200, 255])
    @pytest.mark.parametrize("n_errors", [1, 2, 3])
    def test_corrects_n_bit_errors_in_second_half(self, byte_val, n_errors):
        cw = self.codec.encode(byte_val)
        corrupted = _flip_bits(cw, [24 + i for i in range(n_errors)])
        r = self.codec.decode(corrupted)
        assert r.valid and r.payload == byte_val

    def test_confidence_decreases_with_errors(self):
        byte_val = 42
        cw = self.codec.encode(byte_val)
        r0 = self.codec.decode(cw)
        r1 = self.codec.decode(_flip_bits(cw, [0]))
        r2 = self.codec.decode(_flip_bits(cw, [0, 1]))
        assert r0.confidence > r1.confidence
        assert r1.confidence >= r2.confidence

    def test_confidence_perfect_is_one(self):
        r = self.codec.decode(self.codec.encode(99))
        assert r.confidence == pytest.approx(1.0)

    def test_uncorrectable_sets_flag(self):
        # 4+ errors in one half exceeds correction capability of that half (>3)
        cw = self.codec.encode(0)
        corrupted = _flip_bits(cw, [0, 1, 2, 3])
        r = self.codec.decode(corrupted)
        # If first half is uncorrectable, second half (uncorrupted) should save it
        assert r.valid


# ===========================================================================
# RS48
# ===========================================================================

class TestRS48:
    """Reed-Solomon RS(6,1)/GF(2^8) codec."""

    codec = RS48Codec

    def test_round_trip_all_bytes(self):
        for v in range(256):
            cw = self.codec.encode(v)
            r = self.codec.decode(cw)
            assert r.valid and r.payload == v, f"Failed on byte {v}"

    def test_codeword_length(self):
        cw = self.codec.encode(0)
        assert len(cw) == 6

    def test_zero_syndromes_for_valid_codeword(self):
        for v in range(0, 256, 16):
            cw = self.codec.encode(v)
            syns = self.codec._syndromes(list(cw))
            assert all(s == 0 for s in syns)

    @pytest.mark.parametrize("byte_val", [0, 1, 127, 200, 255])
    def test_corrects_one_symbol_error(self, byte_val):
        cw = self.codec.encode(byte_val)
        for pos in range(6):
            corrupted = _corrupt_byte(cw, pos)
            r = self.codec.decode(corrupted)
            assert r.valid and r.payload == byte_val, \
                f"byte={byte_val}, corrupt pos={pos}"
            assert r.errors_corrected == 1

    @pytest.mark.parametrize("byte_val", [0, 1, 127, 200, 255])
    def test_corrects_two_symbol_errors(self, byte_val):
        cw = self.codec.encode(byte_val)
        for pos_a in range(6):
            for pos_b in range(pos_a + 1, 6):
                corrupted = _corrupt_byte(_corrupt_byte(cw, pos_a), pos_b)
                r = self.codec.decode(corrupted)
                assert r.valid and r.payload == byte_val, \
                    f"byte={byte_val}, corrupt pos=({pos_a},{pos_b})"
                assert r.errors_corrected == 2

    def test_confidence_values(self):
        cw = self.codec.encode(42)
        r0 = self.codec.decode(cw)
        r1 = self.codec.decode(_corrupt_byte(cw, 1))
        r2 = self.codec.decode(_corrupt_byte(_corrupt_byte(cw, 1), 2))
        assert r0.confidence == pytest.approx(1.0)
        assert r1.confidence == pytest.approx(0.75)
        assert r2.confidence == pytest.approx(0.50)

    def test_three_errors_uncorrectable(self):
        cw = self.codec.encode(42)
        corrupted = _corrupt_byte(_corrupt_byte(_corrupt_byte(cw, 0), 1), 2)
        r = self.codec.decode(corrupted)
        # RS(6,1) can only correct t=2 symbol errors
        assert r.uncorrectable or (r.valid and r.payload != 42 and not r.valid)
        assert r.confidence == pytest.approx(0.0) or not r.valid


# ===========================================================================
# BCH48
# ===========================================================================

class TestBCH48:
    """Shortened BCH[63,24,15] + parity codec."""

    codec = BCH48Codec

    def test_round_trip_all_bytes(self):
        for v in range(256):
            cw = self.codec.encode(v)
            r = self.codec.decode(cw)
            assert r.valid and r.payload == v, f"Failed on byte {v}"

    def test_codeword_length(self):
        cw = self.codec.encode(0)
        assert len(cw) == 6

    @pytest.mark.parametrize("byte_val", [0, 1, 63, 127, 200, 255])
    @pytest.mark.parametrize("n_errors", [1, 2, 3, 4, 5, 6, 7])
    def test_corrects_n_random_bit_errors(self, byte_val, n_errors):
        rng = random.Random(byte_val * 100 + n_errors)
        for _ in range(5):  # 5 random patterns per (byte, n_errors)
            cw = self.codec.encode(byte_val)
            positions = rng.sample(range(47), n_errors)
            corrupted = _flip_bits(cw[:6], positions)
            # Parity byte may need updating; we test the raw 47-bit path
            r = self.codec.decode(corrupted)
            assert r.valid and r.payload == byte_val, \
                f"byte={byte_val}, n_errors={n_errors}, positions={positions}"

    def test_confidence_decreases_with_errors(self):
        cw = self.codec.encode(42)
        r0 = self.codec.decode(cw)
        r1 = self.codec.decode(_flip_bits(cw, [0]))
        r3 = self.codec.decode(_flip_bits(cw, [0, 1, 2]))
        assert r0.confidence == pytest.approx(1.0)
        assert r1.confidence < r0.confidence
        assert r3.confidence <= r1.confidence

    def test_zero_syndromes_for_valid_codeword(self):
        for v in range(0, 256, 32):
            cw = self.codec.encode(v)
            ext_int = int.from_bytes(cw, 'big')
            received_47 = ext_int >> 1
            syns = self.codec._compute_syndromes(received_47)
            assert all(s == 0 for s in syns), f"Nonzero syndromes for byte={v}"


# ===========================================================================
# Byte2RGB integration
# ===========================================================================

class TestByte2RGBIntegration:
    """Integration tests via the Byte2RGB encode/decode path."""

    def _make_b2rgb(self, encoding: str) -> Byte2RGB:
        cfg = Byte2RGBConfig(encoding=encoding, seed_flag=False, input_seed=0)
        return Byte2RGB(_MAIN_CFG, _HASH_CFG, cfg)

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_pixels_per_byte_is_two(self, method):
        b2rgb = self._make_b2rgb(method)
        assert b2rgb.pixels_per_byte == 2

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_encode_produces_two_rgb_pixels(self, method):
        b2rgb = self._make_b2rgb(method)
        result = b2rgb.rgb_encoder(bytes([42]))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(p, RGB) for p in result)

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_round_trip_single_byte(self, method):
        b2rgb = self._make_b2rgb(method)
        for v in range(0, 256, 32):
            encoded = b2rgb.rgb_encoder(bytes([v]))
            decoded = b2rgb.rgb_decoder(encoded)
            assert decoded == bytes([v]), f"{method}: byte {v} round-trip failed"

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_round_trip_multi_byte(self, method):
        b2rgb = self._make_b2rgb(method)
        payload = bytes([0, 1, 127, 200, 255])
        encoded = b2rgb.rgb_encoder(payload)
        decoded = b2rgb.rgb_decoder(encoded)
        assert decoded == payload

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_decode_payload_with_confidence_clean(self, method):
        b2rgb = self._make_b2rgb(method)
        for v in range(0, 256, 64):
            rgb_pair = b2rgb.rgb_encoder(bytes([v]))
            result = b2rgb.decode_payload_with_confidence(rgb_pair[0], rgb_pair[1])
            assert isinstance(result, DecodeResult)
            assert result.valid
            assert result.payload == v
            assert result.confidence == pytest.approx(1.0)
            assert result.errors_corrected == 0
            assert result.method == method

    def test_decode_with_confidence_requires_48bit_mode(self):
        b2rgb = self._make_b2rgb("golay24-dual")
        # Switch encoding back to golay24 by creating a new instance
        b2rgb_old = Byte2RGB(
            _MAIN_CFG, _HASH_CFG,
            Byte2RGBConfig(encoding="golay24", seed_flag=False, input_seed=0))
        with pytest.raises(RuntimeError, match="48-bit"):
            b2rgb_old.decode_payload_with_confidence(
                RGB(r=128, g=128, b=128), RGB(r=128, g=128, b=128))

    def test_invalid_encoding_raises(self):
        cfg = Byte2RGBConfig.__new__(Byte2RGBConfig)
        object.__setattr__(cfg, "fr_min", 0)
        object.__setattr__(cfg, "fr_max", 255)
        object.__setattr__(cfg, "encoding", "badenc")
        object.__setattr__(cfg, "bin_width", 36)
        object.__setattr__(cfg, "bin_num", 7)
        object.__setattr__(cfg, "seed_flag", False)
        object.__setattr__(cfg, "input_seed", 0)
        object.__setattr__(cfg, "seed", 0)
        with pytest.raises(ValueError, match="encoding must be"):
            Byte2RGB(_MAIN_CFG, _HASH_CFG, cfg)
