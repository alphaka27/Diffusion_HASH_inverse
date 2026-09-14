"""Reversible message representations for diffusion experiments."""

from .bgv import (
    BGVConfig,
    BGVDecoder,
    BGVEncoder,
    DecodeResult,
    bits_to_byte,
    byte_to_bits,
    decode_byte_glyph,
    encode_byte_glyph,
    plot_bgv,
    save_bgv_image,
)

__all__ = [
    "BGVConfig",
    "BGVDecoder",
    "BGVEncoder",
    "DecodeResult",
    "bits_to_byte",
    "byte_to_bits",
    "decode_byte_glyph",
    "encode_byte_glyph",
    "plot_bgv",
    "save_bgv_image",
]
