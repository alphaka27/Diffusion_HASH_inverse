"""
Docstring for diffusion_hash_inv.utils.drawer
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from diffusion_hash_inv.core import Byte2RGB, RGB
from diffusion_hash_inv.core import Logs

class Drawer:
    """
    A utility class for drawing images with specific color.
    """

    def __init__(self, outer: int, inner: int):
        """
        Initialize the Drawer with specified Outer Length (outer) and Inner Diameter (inner).

        Args:
            outer (int): Outer Length.
            inner (int): Inner Diameter.
        """
        assert outer > 0, "Outer Length must be positive."
        assert inner > 0, "Inner Diameter must be positive."
        assert outer > inner, "Outer Length must be greater than Inner Diameter."

        self.od = outer
        self.id = inner

    def draw_image(self, color: tuple) -> Image.Image:
        """
        Draw an image with the specified color.

        Args:
            color (tuple): A tuple representing the RGB color (R, G, B).

        Returns:
            Image.Image: The drawn image.
        """
        img = Image.new("RGB", (self.od, self.od), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw outer circle
        draw.ellipse(
            [(0, 0), (self.od - 1, self.od - 1)],
            fill=color,
            outline=(0, 0, 0),
        )

        # Draw inner circle
        offset = (self.od - self.id) // 2
        draw.ellipse(
            [(offset, offset), (self.od - offset - 1, self.od - offset - 1)],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
        )

        return img

if __name__ == "__main__":
    drawer = Drawer(100, 60)
    color = Byte2RGB(128)
    img = drawer.draw_image(color)
    img.show()
