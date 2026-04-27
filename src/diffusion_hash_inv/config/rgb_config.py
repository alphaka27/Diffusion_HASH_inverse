"""
Configuration for RGB image processing in diffusion hash inversion.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
import secrets

@dataclass
class Byte2RGBConfig:
    """
    Configuration for Byte to RGB conversion.
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
            f"  bin_width: {self.bin_width},\n"
            f"  bin_num: {self.bin_num},\n"
            f"  seed_flag: {self.seed_flag},\n"
            f"  seed: {self.seed}\n")

    @staticmethod
    def help() -> str:
        """
        Return a help string describing the Byte2RGBConfig fields.
        """
        return (
            "Byte2RGBConfig\n"
            "  fr_min: Minimum byte value (inclusive) for RGB conversion.\n"
            "  fr_max: Maximum byte value (inclusive) for RGB conversion.\n"
            "  bin_width: Width of each bin for RGB conversion.\n"
            "  bin_num: Number of bins for RGB conversion.\n"
            "  seed_flag: Whether to use a random seed for RGB conversion.\n"
            "    if True, a random seed will be generated. If False, input_seed will be used.\n"
            "  input_seed: Optional seed value if seed_flag is False.\n"
            "  seed: The actual seed value used for RGB conversion.\n")

@dataclass
class ImgConfig:
    """
    Configuration for RGB image processing.
    """

    img_size: Tuple[int, int] = (28, 28) # Width, Height
    center_size: Tuple[int, int] = (14, 14) # Width, Height

    def __repr__(self):
        return (
            "ImgConfig\n"
            f"  img_size={self.img_size},\n"
            f"  center_size={self.center_size}")
