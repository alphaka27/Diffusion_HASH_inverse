"""
Configuration for RGB image processing in diffusion hash inversion.
"""
from dataclasses import dataclass

@dataclass
class Byte2RGBConfig:
    """
    Configuration for Byte to RGB conversion.
    """

    fr_min: int = 0
    fr_max: int = 255
    bin_width: int = 36
    bin_num: int = 7
