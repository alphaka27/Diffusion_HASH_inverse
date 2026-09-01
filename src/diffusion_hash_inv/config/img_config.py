"""
Configuration Image Parameter
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import math

@dataclass(frozen=True)
class ImageConfig:
    """
    Configuration for image parameters.
    Attributes:
        size (Optional[List[int]]): Size of the image as [width, height].
        grid_size (Optional[List[int]]): Size of each grid as [width, height].
        encoding_method (str): Encoding method to use ("grid", "cube", or "cuboid").
        grid (List[int]): Grid size for encoding, either [2, 2] or [3, 3].
    """
    size: Optional[List[int, int]] = None # size of the image.
    grid_size: Optional[List[int, int]] = None # size of each grid.
    encoding_method: str = "grid" # grid, cube, cuboid
    grid: List[int, int] = \
        field(default_factory=lambda: [2, 2]) # grid size for encoding. 2X2 or 3X3
    radius: int = 0

    def __post_init__(self):
        if self.encoding_method not in ["grid", "cube", "cuboid"]:
            raise ValueError(
                "Invalid encoding method: "
                f"{self.encoding_method}. Must be 'grid', 'cube', or 'cuboid'.")
        if self.size is not None and len(self.size) != 2:
            raise ValueError(f"Size must be a list of two integers, got {self.size}.")
        if self.grid_size is None or len(self.grid_size) != 2:
            raise ValueError(f"Grid size must be a list of two integers, got {self.grid_size}.")
        if len(self.grid) != 2:
            raise ValueError(f"Grid must be a list of two integers, got {self.grid}.")
        if self.grid not in ([2, 2], [3, 3]):
            raise ValueError(f"Grid must be either [2, 2] or [3, 3], got {self.grid}.")

        if self.size is None:
            w = self.grid_size[0] * self.grid[0]
            h = self.grid_size[1] * self.grid[1]
            object.__setattr__(self, 'size', [w, h])

    def __repr__(self):
        return (
            "Image Configuration:\n"
            f"ImageConfig(size={self.size}, \ngrid_size={self.grid_size}, \n"
            f"encoding_method='{self.encoding_method}', \ngrid={self.grid})\n")

    def __call__(self):
        self.grid_size = self.grid_size if self.grid_size is not None else [10, 10]
        self.grid = self.grid if self.grid is not None else [2, 2]
        self.size = \
            [self.grid_size[0] * self.grid[0], \
             self.grid_size[1] * self.grid[1]]
        self.encoding_method = self.encoding_method if self.encoding_method is not None else "grid"

    def grid_validate(self, candidate: List[Any]):
        """
        Validate the grid configuration.
        """
        _len = len(candidate)
        if _len < math.pow(2, 2**2):
            object.__setattr__(self, 'grid', [2, 2])
        elif _len < math.pow(2, 3**2):
            object.__setattr__(self, 'grid', [3, 3])
        else:
            raise ValueError("The given candidate list is too large for the Grid size. "
                            f"Given candidate list length: {_len}.")
        w = self.grid_size[0] * self.grid[0]
        h = self.grid_size[1] * self.grid[1]
        object.__setattr__(self, 'size', [w, h])
