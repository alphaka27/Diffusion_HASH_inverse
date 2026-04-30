"""
Make RGB images from Logs.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from tqdm import tqdm

import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, DataLoader

from diffusion_hash_inv.config import ImgConfig
from diffusion_hash_inv.config import Byte2RGBConfig
from diffusion_hash_inv.core import RGB, RGBA
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.validation.encoding_validation import encoding_validate
from diffusion_hash_inv.utils.byte2rgb import Byte2RGB
from diffusion_hash_inv.utils.file_io import FileIO
from diffusion_hash_inv.main.context import RuntimeConfig

class RGBImgMaker:
    """
    A class to make RGB images from Logs.
    """

    def __init__(self, runtime_cfg: RuntimeConfig,
                io_controller: FileIO,
                rgb_config: Byte2RGBConfig):
        self.runtime_cfg = runtime_cfg
        self.main_cfg = runtime_cfg.main
        self.hash_cfg = runtime_cfg.hash
        self.io_controller = io_controller
        self.byte2rgb = Byte2RGB(main_config=self.main_cfg,
                                hash_config=self.hash_cfg,
                                rgb_config=rgb_config)
        self.log_hierarchy: Optional[List[str]] = []
        print("RGB Image Maker Initialized.")


    def _image_concater(self, images: List[Image.Image], direction: str) -> Image.Image:
        """
        Concatenate a list of images into a single image.
        """
        assert len(images) > 0, "No images to concatenate."
        assert direction in ["vertical", "horizontal"], \
            "Direction must be 'vertical' or 'horizontal'."

        imgs = [img.convert("RGBA") for img in images]
        pivot = (images[0].width, images[0].height)
        for img in imgs:
            _hori_cond = pivot[0] % img.width == 0 and img.height == pivot[1]
            _vert_cond = img.width == pivot[0] and pivot[1] % img.height == 0
            assert (_vert_cond and direction == "vertical") or \
                (_hori_cond and direction == "horizontal"), \
                f"All images must have the same dimensions for concatenation.\n" \
                f"Expected: {pivot}, Got: ({img.width}, {img.height})"

        width = max(img.width for img in imgs) \
            if direction == "vertical" else sum(img.width for img in imgs)
        height = sum(img.height for img in imgs) \
            if direction == "vertical" else max(img.height for img in imgs)
        new_img = Image.new("RGBA", (width, height))
        offset = 0
        for img in imgs:
            new_img.paste(img, (0, offset) if direction == "vertical" else (offset, 0))
            offset += img.height if direction == "vertical" else img.width
        return new_img


    def _image_formatter(self, \
                        rgb_data: Tuple[RGB] | Tuple[RGBA], \
                        image_size: Tuple[int, int], \
                        center_size: Tuple[int, int]) -> Image.Image:
        """
        Make RGB image from RGB data.
        """
        assert image_size[0] > 0 and image_size[1] > 0, "Image size must be positive."
        assert center_size[0] > 0 and center_size[1] > 0, "Center size must be positive."
        assert center_size[0] <= image_size[0] and center_size[1] <= image_size[1], \
            "Center size must be smaller than or equal to image size."

        background_color = (255, 255, 255, 255)  # White with full opacity
        center_x = (image_size[0] - center_size[0]) // 2
        center_y = (image_size[1] - center_size[1]) // 2
        assert center_x >= 0 and center_y >= 0, "Center size must be smaller than image size."

        assert len(rgb_data) > 0, "RGB data cannot be empty."
        assert all(isinstance(pixel, (RGB, RGBA)) for pixel in rgb_data), \
            "All items in rgb_data must be of type RGB or RGBA. " \
            f"Got types: {[type(pixel) for pixel in rgb_data]}" \
            f" with values: {rgb_data}"
        assert image_size[0] >= center_size[0] + center_x and \
            image_size[1] >= center_size[1] + center_y, \
            "Image size must be large enough to accommodate center size with offset."

        frames: List[Image.Image] = []
        for rgb in rgb_data:
            canvas = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
            canvas[:, :] = background_color
            canvas[center_y:center_y + center_size[1], center_x:center_x + center_size[0]] = \
                (rgb.r, rgb.g, rgb.b, 255) if isinstance(rgb, RGB) else (rgb.r, rgb.g, rgb.b, rgb.a)
            frames.append(Image.fromarray(canvas, "RGBA"))

        assert len(frames) > 0, "Failed to create image from RGB data."
        if len(frames) == 1:
            return frames[0]
        return self._image_concater(frames, direction="horizontal")


    def image_formatter(self, \
                        rgb_data: Tuple[RGB] | List[Tuple[RGB]] | Tuple[RGBA] | List[Tuple[RGBA]]) \
                        -> Image.Image:
        """
        Make RGB image from RGB data.
        """
        assert len(rgb_data) > 0, "RGB data cannot be empty."
        img_size = (ImgConfig().img_size[0], ImgConfig().img_size[1])  # Width, Height
        center_size = (ImgConfig().center_size[0], ImgConfig().center_size[1])  # Width, Height

        if isinstance(rgb_data, Tuple):
            if isinstance(rgb_data[0], (RGB, RGBA)):
                return self._image_formatter(rgb_data, img_size, center_size)

            if isinstance(rgb_data[0], Tuple):
                ret = None
                for data in rgb_data:
                    if not isinstance(data, Tuple):
                        raise ValueError(
                            "All items in rgb_data tuple must be of type Tuple[RGB] or Tuple[RGBA]"
                            )
                    img = self._image_formatter(data, img_size, center_size)
                    if ret is None:
                        ret = img
                    else:
                        ret = self._image_concater([ret, img], direction="vertical")
                assert ret is not None, "Failed to create image from RGB data."
                return ret

            raise ValueError(
                "All items in rgb_data tuple must be of type Tuple[RGB] or Tuple[RGBA]"
                )

        raise ValueError("rgb_data must be a tuple of RGB or RGBA tuples.")

    def data_encoder(self, data: str | bytes) \
        -> Tuple[RGB] | Tuple[RGBA]:
        """
        Encode data to RGB or RGBA format.
        """
        if isinstance(data, (str, bytes)):
            ret = self.byte2rgb.rgb_encoder(data)
            if self.main_cfg.debug_flag:
                success = encoding_validate(data, ret, self.byte2rgb)
            else:
                success = True  # Skip validation in non-debug mode for performance

            if success:
                return ret

            raise RuntimeError(f"Encoding validation failed for data: {data}\n"
                            f"Encoded RGB: {ret}\n"
                            f"Original data: {data}\n"
                            f"Decoded data: {self.byte2rgb.rgb_decoder(ret)}\n"
                            f"Success: {success}")

        if isinstance(data, list):
            ret = []
            for item in data:
                if not isinstance(item, (str, bytes)):
                    raise ValueError("All items in data list must be of type str or bytes.")
                encoded_item = self.byte2rgb.rgb_encoder(item)
                if self.main_cfg.debug_flag:
                    success = encoding_validate(item, encoded_item, self.byte2rgb)
                else:
                    success = True  # Skip validation in non-debug mode for performance
                if not success:
                    raise RuntimeError(f"Encoding validation failed for item: {item}\n"
                                    f"Encoded RGB: {encoded_item}\n"
                                    f"Original item: {item}\n"
                                    f"Decoded item: {self.byte2rgb.rgb_decoder(encoded_item)}\n"
                                    f"Success: {success}")
                ret.append(encoded_item)
            return tuple(ret)

        raise ValueError("Unsupported data type for encoding.")

    def img_writer(self, log_dict: List[Dict]) -> None:
        """
        Write RGB image data to file.
        """
        filename, message, step_logs = Logs.log_parser(log_dict)

        parsed_logs = Logs.steplogs_parser(step_logs, self.log_hierarchy)

        encoded_message = self.data_encoder(message)
        rgb_message = self.image_formatter(encoded_message)

        self.io_controller.file_writer("message.png",
                                    rgb_message,
                                    parent_dir=filename,
                                    data_type="data")

        for log in parsed_logs:
            assert isinstance(log, dict), "Parsed log must be a dictionary."
            path = list(log.keys())
            assert len(path) == 1, \
                f"Parsed log dictionary must have exactly one key. {len(path)} keys found."
            path = path[0]
            data = log[path]
            _file_name = path.split("/")[-1]
            assert isinstance(data, (str, int, float, list, tuple, bytes)), \
                "Parsed log data must be of type str, int, float, list, tuple, or bytes."
            encoded_log = None
            encoded_log = self.data_encoder(data)
            if self.main_cfg.verbose_flag:
                print(encoded_log)

            path = "/".join(path.split("/")[:-1])
            path = Path(path)
            rgb_log = self.image_formatter(encoded_log)
            self.io_controller.file_writer(f"{_file_name}.png",
                                        rgb_log,
                                        parent_dir=Path(filename, path),
                                        data_type="data")


    def main(self) -> None:
        """
        Main method to convert bytes data to a list of RGB tuples.
        """
        logs = self.io_controller.\
            get_latest_files_by_date(self.hash_cfg.hash_alg, self.hash_cfg.length)
        print(f"Found {len(logs)} logs to process.")


        log_process = tqdm(
            Logs.iter_logs_with_hierarchy(self.io_controller, self.log_hierarchy, logs),
            total=len(logs), desc="Processing Logs", unit="log", position=0)

        processing_info = tqdm(total=0, desc="Processing Info", bar_format="{desc}", position=1)

        for log_dict in log_process:
            self.img_writer(log_dict)
            _key = list(log_dict.keys())
            if len(_key) == 1:
                _key = _key[0]
            else:
                raise ValueError("Multiple keys found in log_dict.")
            processing_info.set_description_str(f"Processed log: {_key}")


class EMNISTImgMaker:
    """
    A class to make Images from EMNIST dataset.
    """
    def __init__(self, runtime_cfg: RuntimeConfig,
                io_controller: FileIO,
                target_classes: Optional[List[str]] = None):
        self.runtime_cfg = runtime_cfg
        self.main_cfg = runtime_cfg.main
        self.hash_cfg = runtime_cfg.hash
        self.io_controller = io_controller
        self.target_classes = target_classes

        print("EMNIST Image Maker Initialized.")

    def load_emnist_data(self, file_path: Optional[Path] = None) -> ConcatDataset:
        """
        Load EMNIST data from the given file path.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if file_path is None:
            file_path = Path(self.runtime_cfg.output.root_dir, "EMNIST_data")
        train_dataset = datasets.EMNIST(root=file_path, split='byclass', download=True
                                        , transform=transform, train=True)
        test_dataset = datasets.EMNIST(root=file_path, split='byclass', download=True
                                        , transform=transform, train=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])

        return full_dataset

    def emnist_dataloader(self, dataset: ConcatDataset, batch_size: int = 64) -> DataLoader:
        """
        Create a dataloader for the EMNIST dataset.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return dataloader


class HDF5Maker:
    """
    A class to make HDF5 files from Logs.
    """
    def __init__(self, runtime_cfg: RuntimeConfig,
                io_controller: FileIO):
        self.runtime_cfg = runtime_cfg
        self.main_cfg = runtime_cfg.main
        self.hash_cfg = runtime_cfg.hash
        self.io_controller = io_controller

        print("HDF5 Maker Initialized.")


class ImageMaker(RGBImgMaker, EMNISTImgMaker, HDF5Maker):
    """
    A class to make images from Logs.
    """
    def __init__(self, runtime_cfg: RuntimeConfig,
                io_controller: FileIO,
                rgb_config: Byte2RGBConfig):
        RGBImgMaker.__init__(self, runtime_cfg, io_controller, rgb_config)
        EMNISTImgMaker.__init__(self, runtime_cfg, io_controller)
        HDF5Maker.__init__(self, runtime_cfg, io_controller)

        print("Image Maker Initialized.")
