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
    bytes as extended Golay-protected bit-position RGB pixels.  ``fr_min`` and
    ``fr_max`` define the channel range to split.  ``bin_width`` and
    ``bin_num`` are retained for legacy config compatibility but are not used
    by the Golay encoder.
    """

    fr_min: int = 0
    fr_max: int = 255
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
