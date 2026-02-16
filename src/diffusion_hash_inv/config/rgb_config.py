"""
Configuration for RGB image processing in diffusion hash inversion.
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Byte2RGBConfig:
    """
    Configuration for Byte to RGB conversion.
    """

    fr_min: int = 0
    fr_max: int = 255
    bin_width: int = 36
    bin_num: int = 7

@dataclass
class ImgConfig:
    """
    Configuration for RGB image processing.
    """

    img_size: Tuple[int, int] = (32, 32) # Width, Height
    center_size: Tuple[int, int] = (16, 16) # Width, Height
