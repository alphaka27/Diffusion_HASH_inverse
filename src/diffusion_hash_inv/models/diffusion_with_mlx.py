"""
Diffusion Model toy example with MLX
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import mlx.core as mx

METAL_OK = mx.is_metal_available()

class DiffusionWithMLX:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if not METAL_OK:
            raise RuntimeError("Metal is not available on this system.")
        self.model = mx.load_model(self.model_path)

    def predict(self, input_image: Image.Image) -> Image.Image:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        # Preprocess the input image
        input_array = np.array(input_image).astype(np.float32) / 255.0
        input_tensor = mx.Tensor(input_array)

        # Run the model prediction
        output_tensor = self.model(input_tensor)

        # Postprocess the output tensor to an image
        output_array = (output_tensor.numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(output_array)

        return output_image
