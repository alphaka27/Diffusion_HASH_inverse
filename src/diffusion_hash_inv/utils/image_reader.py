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
from diffusion_hash_inv.utils import FileIO


