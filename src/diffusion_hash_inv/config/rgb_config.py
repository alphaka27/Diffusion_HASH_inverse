"""
Configuration for RGB image processing in diffusion hash inversion.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
import secrets

@dataclass
class Byte2RGBConfig:
    """
    Configuration for byte-to-RGB conversion.

    The current encoder splits each RGB channel into two halves and stores
    bytes as error-corrected 48-bit RGB pairs by default.  ``fr_min`` and
    ``fr_max`` define the channel range for the legacy Golay bit-position
    encoder.  ``bin_width`` and ``bin_num`` are retained for legacy config
    compatibility but are not used by the 48-bit pair encoders.

    ``encoding`` selects the active encoder:
    - ``"linear48"``: Binary linear [48,8,17] ECC — **2 pixels per byte**, 8-bit error correction.
    - ``"golay24"``: Extended Golay(24,12) ECC — 24 pixels per byte, 3-bit error correction.
    - ``"legacy-bin"``: Direct RGB bin mapping — 1 pixel per byte, no error correction.
    - ``"cube-id"``: RGB cube-id mapping from Encoding Method 2 — 1 pixel per byte, no error correction.
    - ``"golay24-dual"``: 2× Extended Golay(24,12) — **2 pixels per byte**, 6-bit error correction.
    - ``"rs48"``: Reed-Solomon RS(6,1)/GF(2^8) — **2 pixels per byte**, 2 byte-error correction.
    - ``"bch48"``: Shortened BCH[63,24,15] — **2 pixels per byte**, 7-bit error correction.
    """

    fr_min: int = 0
    fr_max: int = 255
    encoding: str = "linear48"
    bin_width: int = 36
    bin_num: int = 7
    seed_flag: bool = True
    input_seed: Optional[int] = None
    seed: int = field(default=0, init=False)

    def __post_init__(self):
        if self.seed_flag:
            object.__setattr__(self, "seed", secrets.randbits(32))
        else:
            assert self.input_seed is not None, "input_seed must be provided if seed_flag is False"
            object.__setattr__(self, "seed", 0 if self.input_seed is None else self.input_seed)

    def __repr__(self):
        return (
            "Byte2RGBConfig\n"
            f"  fr_min: {self.fr_min},\n"
            f"  fr_max: {self.fr_max},\n"
            f"  encoding: {self.encoding},\n"
            f"  bin_width: {self.bin_width} (legacy, unused by Golay encoder),\n"
            f"  bin_num: {self.bin_num} (legacy, unused by Golay encoder),\n"
            f"  seed_flag: {self.seed_flag},\n"
            f"  seed: {self.seed}\n")

    @staticmethod
    def help() -> str:
        """
        Return a help string describing the Byte2RGBConfig fields.
        """
        return (
            "Byte2RGBConfig\n"
            "  fr_min: Minimum RGB channel value (inclusive) for Golay conversion.\n"
            "  fr_max: Maximum RGB channel value (inclusive) for Golay conversion.\n"
            "  encoding: Encoder type:\n"
            "    'linear48'     —  2px/byte, 8-bit error correction (binary [48,8,17])\n"
            "    'golay24'      — 24px/byte, 3-bit error correction (Extended Golay)\n"
            "    'legacy-bin'   —  1px/byte, no error correction\n"
            "    'cube-id'      —  1px/byte, Encoding Method 2 RGB cube-id mapping\n"
            "    'golay24-dual' —  2px/byte, 6-bit error correction (2× Golay, 48-bit)\n"
            "    'rs48'         —  2px/byte, 2-byte error correction (RS(6,1)/GF(2^8))\n"
            "    'bch48'        —  2px/byte, 7-bit error correction (BCH[63,24,15] shortened)\n"
            "  bin_width: Legacy byte-bin width, unused by the Golay encoder.\n"
            "  bin_num: Legacy byte-bin count, unused by the Golay encoder.\n"
            "  seed_flag: Legacy seed switch, retained for compatibility.\n"
            "  input_seed: Optional legacy seed value if seed_flag is False.\n"
            "  seed: The actual legacy seed value stored in the config.\n")

@dataclass
class ImgConfig:
    """
    Configuration for RGB image processing.
    """

    img_size: Tuple[int, int] = (28, 28) # Width, Height
    center_size: Tuple[int, int] = (28, 28) # Width, Height

    def __repr__(self):
        return (
            "ImgConfig\n"
            f"  img_size={self.img_size},\n"
            f"  center_size={self.center_size}")
