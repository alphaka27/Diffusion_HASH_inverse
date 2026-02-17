"""
Docstring for diffusion_hash_inv.utils.image_reader
"""

from __future__ import annotations
from PIL import Image
from pathlib import Path


from diffusion_hash_inv.core import RGB
from diffusion_hash_inv.utils import FileIO

class ImageReader:
    """
    A class to read RGB values from an image file.
    """

    def __init__(self, io_controller: FileIO):
        self.io_controller = io_controller

    def read_image(self, image_path: str | Path) -> list[RGB]:
        """
        Read an image and return a list of RGB tuples.

        Args:
            image_path (str | Path): The path to the image file.
        Returns:
            list[RGB]: A list of RGB tuples representing the pixel values of the image.
        """
        
