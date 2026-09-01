"""
Encoding module for the diffusion hash inversion model.
Encoding with proper encoding and decoding methods for bytes.
"""

from diffusion_hash_inv.config import RuntimeConfig
from diffusion_hash_inv.encoding.rgb_grid import RGBGridEncoder
# from diffuision_hash_inv.encoding.rgb_cube import RGBCubeEncoder
# from diffusion_hash_inv.encoding.rgb_cuboid import RGBCuboidEncoder

class Encoder:
    """
    Encoder class for encoding and decoding byte data using different encoding methods.
    """

    def __init__(self, config: RuntimeConfig):
        self.img_cfg = config.image
        self.encoder = self._select_encoder()

    def _select_encoder(self):
        """
        Select the appropriate encoder based on the configuration.

        Returns:
            An instance of the selected encoder class.
        """
        if self.img_cfg.encoding_method == "grid":
            return RGBGridEncoder(self.img_cfg)
        if self.img_cfg.encoding_method == "cube":
            raise NotImplementedError("Not Implemented Yet")
            # return RGBCubeEncoder(self.img_cfg)
        if self.img_cfg.encoding_method == "cuboid":
            raise NotImplementedError("Not Implemented Yet")
            # return RGBCuboidEncoder(self.img_cfg)

        raise ValueError(f"Unsupported encoding method: {self.img_cfg.encoding_method}")

    def encode(self, byte_data: bytearray):
        """
        Encode the given byte data into an image.

        Args:
            byte_data (bytearray): The input byte data to encode.

        Returns:
            Image.Image: The encoded image.
        """
        return self.encoder.encode(byte_data)

    def decode(self, image):
        """
        Decode the given image back into byte data.

        Args:
            image (Image.Image): The input image to decode.

        Returns:
            bytes: The decoded byte data.
        """
        return self.encoder.decode(image)
