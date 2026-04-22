"""
Docstring for diffusion_hash_inv.utils.image_reader

Reads generated hash-step images and extracts the center color
from each block, returning a structured list of RGB values.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from pathlib import Path
import numpy as np

from diffusion_hash_inv.core import RGB, RGBA
from diffusion_hash_inv.config import ImgConfig
from diffusion_hash_inv import FileIO


class ImageReader:
    """
    Reads generated hash-step images and extracts the center color
    from each block, returning a structured list of RGB values.
    """

    def __init__(self, config: ImgConfig):
        self.config = config

    def read_image(self, image_path: Path) -> List[List[RGB]]:
        """
        Reads an image and extracts the center color from each block.

        Args:
            image_path (Path): Path to the image file.
        Returns:
            List[List[RGB]]: A 2D list of RGB values representing the center color
                            of each block in the image.
        """
        image = Image.open(image_path).convert("RGBA")
        img_array = np.array(image)

        block_size = self.config.block_size
        height, width, _ = img_array.shape

        rgb_values = []
        for y in range(0, height, block_size):
            row = []
            for x in range(0, width, block_size):
                center_x = x + block_size // 2
                center_y = y + block_size // 2
                if center_x < width and center_y < height:
                    rgba_pixel = img_array[center_y, center_x]
                    rgb_pixel = tuple(rgba_pixel[:3])  # Ignore alpha channel
                    row.append(rgb_pixel)
            rgb_values.append(row)

        return rgb_values