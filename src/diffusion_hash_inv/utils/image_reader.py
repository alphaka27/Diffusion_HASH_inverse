"""
Docstring for diffusion_hash_inv.utils.image_reader

Reads generated hash-step images and extracts the center color
from each block, returning a structured list of RGB values.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np

from diffusion_hash_inv.core import RGB, RGBA
from diffusion_hash_inv.config import ImgConfig
from diffusion_hash_inv.utils.file_io import FileIO


class ImageReader:
    """
    ImageReader is responsible for reading generated hash-step images and extracting
    the center color from each block, returning a structured list of RGB values.
    """

    def __init__(self, img_config: ImgConfig):
        self.img_config = img_config

    def get_image(self, image_path: Path) -> Image.Image:
        """
        Reads an image from the specified path.

        Args:
            image_path (Path): The path to the image file.
        Returns:
            Image.Image: The loaded image.
        """
        return Image.open(image_path)

    def _image_parser(self, img: Image.Image) -> List[List[RGB]]:
        """
        Parses an image and extracts first line of image

        Args:
            img (Image.Image): The image to parse.
        Returns:
            List[List[RGB]]: A 2D list of RGB values representing the center color of each block.
        """
        image_array = np.array(img)
        block_size = self.img_config.block_size
        blocks_per_row = self.img_config.blocks_per_row
        blocks_per_col = self.img_config.blocks_per_col

        rgb_values = []
        for row in range(blocks_per_col):
            row_values = []
            for col in range(blocks_per_row):
                center_x = col * block_size + block_size // 2
                center_y = row * block_size + block_size // 2
                rgba_value = image_array[center_y, center_x]
                rgb_value = RGB(rgba_value[0], rgba_value[1], rgba_value[2])
                row_values.append(rgb_value)
            rgb_values.append(row_values)

        return rgb_values

    def read_image(self, image_path: Path) -> List[List[RGB]]:
        """
        Reads an image and extracts the center color from each block.

        Args:
            image_path (Path): The path to the image file.
        Returns:
            List[List[RGB]]: A 2D list of RGB values representing the center color of each block.
        """
        image = Image.open(image_path).convert("RGBA")
        image_array = np.array(image)
        block_size = self.img_config.block_size
        blocks_per_row = self.img_config.blocks_per_row
        blocks_per_col = self.img_config.blocks_per_col

        rgb_values = []
        for row in range(blocks_per_col):
            row_values = []
            for col in range(blocks_per_row):
                center_x = col * block_size + block_size // 2
                center_y = row * block_size + block_size // 2
                rgba_value = image_array[center_y, center_x]
                rgb_value = RGB(rgba_value[0], rgba_value[1], rgba_value[2])
                row_values.append(rgb_value)
            rgb_values.append(row_values)

        return rgb_values
