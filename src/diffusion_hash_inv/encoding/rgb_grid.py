"""
Encoding Bytes to Grid-structured image 
"""

from PIL import Image
from typing import List, Tuple, TYPING
import numpy as np


from diffusion_hash_inv.config \
    import MainConfig, HashConfig, MessageConfig, OutputConfig, ImageConfig, RuntimeConfig

if TYPING:
    from diffusion_hash_inv.type import RGB


class RGBGridEncoder:
    """
    Encoder for converting bytes to a grid-structured RGB image.
    """

    def __init__(self, config: RuntimeConfig):
        self.runtime_cfg = config
        self.img_cfg = config.image
        self.msg_cfg = config.message
        self.img_cfg.grid_validate(self.msg_cfg.candidate_list)

    def _digit_converter(self, data: bytes):
        if not 0 <= data <= 9:
            raise ValueError("Data must be a digits")

    def _zero(self) -> np.ndarray:
        """
        Encoding bit to Zero
        """

        w = self.img_cfg.grid_size[0]
        h = self.img_cfg.grid_size[1]
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _ = bg

    def _one(self, color: RGB) -> np.ndarray:
        """
        Encoding bit to One
        """

        w = self.img_cfg.grid_size[0]
        h = self.img_cfg.grid_size[1]
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :, 0] = color.r
        bg[:, :, 1] = color.g
        bg[:, :, 2] = color.b
        fg = np.ones((h, w, 3), dtype=np.uint8) * 255

    def encode(self, byte_data: bytes) -> Image.Image:
        """
        Encode the given byte data into a grid-structured RGB image.

        Args:
            byte_data (bytes): The input byte data to encode.

        Returns:
            Image.Image: The encoded grid-structured RGB image.
        """
        bin_data = bin(byte_data)[2:] # convert input data to binary



    def decode(self, image: Image.Image) -> bytes:
        """
        Decode the given grid-structured RGB image back into byte data.

        Args:
            image (Image.Image): The input grid-structured RGB image to decode.

        Returns:
            bytes: The decoded byte data.
        """
