"""
Make RGB images from Logs.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from tqdm import tqdm

import numpy as np
from PIL import Image


from diffusion_hash_inv.config import MainConfig, HashConfig, ImgConfig
from diffusion_hash_inv.config import Byte2RGBConfig
from diffusion_hash_inv.core import RGB, RGBA
from diffusion_hash_inv.logger import Logs
from diffusion_hash_inv.validation.encoding_validation import encoding_validate
from diffusion_hash_inv.utils.byte2rgb import Byte2RGB
from diffusion_hash_inv.utils.file_io import FileIO

class RGBImgMaker:
    """
    A class to make RGB images from Logs.
    """

    def __init__(self, main_cfg: MainConfig,
                hash_cfg: HashConfig,
                io_controller: FileIO,
                rgb_config: Byte2RGBConfig):
        self.main_cfg = main_cfg
        self.hash_cfg = hash_cfg
        self.io_controller = io_controller
        self.byte2rgb = Byte2RGB(main_config=self.main_cfg,
                                hash_config=self.hash_cfg,
                                rgb_config=rgb_config)
        self.log_hierarchy: Optional[Dict[str, Any]] = None


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
        assert image_size[0] > center_size[0] + center_x and \
            image_size[1] > center_size[1] + center_y, \
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
            success = encoding_validate(data, ret, self.byte2rgb)

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
                success = encoding_validate(item, encoded_item, self.byte2rgb)
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
        filename, message, step_logs = self.log_parser(log_dict)

        parsed_logs = self.steplogs_parser(step_logs)

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
            if path == "3rd Step" and self.main_cfg.debug_flag:
                breakpoint()
            path = "/".join(path.split("/")[:-1])
            path = Path(path)
            rgb_log = self.image_formatter(encoded_log)
            self.io_controller.file_writer(f"{_file_name}.png",
                                        rgb_log,
                                        parent_dir=Path(filename, path),
                                        data_type="data")

    def _dfs_searcher(self, data_dict: Dict[str, Any], key_path: Path = None) \
        -> List[Dict[str, Any]]:
        """
        Depth-first search to traverse the log hierarchy.
        """
        ret = None
        assert isinstance(data_dict, dict), "Data must be a dictionary."
        for key, value in data_dict.items():
            current_path = key_path / key if key_path is not None else Path(key)
            if isinstance(value, (str, int, float, list, tuple, bytes)):
                if ret is None:
                    ret = [{str(current_path): value},]
                else:
                    ret += [{str(current_path): value},]
            elif isinstance(value, dict):
                has_match = any(h in k for k in value for h in self.log_hierarchy)

                if not has_match:
                    _temp = [v for v in value.values() \
                            if isinstance(v, (str, int, float, list, tuple, bytes))]
                    _ret = [{str(current_path): _temp},] if len(_temp) > 0 else None
                else:
                    _ret = self._dfs_searcher(value, current_path)

                if _ret is not None:
                    ret = _ret if ret is None else ret + _ret

            else:
                raise ValueError(
                    f"Unsupported data type in log hierarchy: {type(value)} at path: {current_path}"
                    )
        return ret if ret is not None else []


    def steplogs_parser(self, step_logs: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
        """
        Parse step logs to extract log information.
        'Logs' field contains logs for each step.
        'Logs' must be a dictionary with step names as keys.

        Returns:
            A tuple containing the parsed log information.  
            **key**: path to the log in the hierarchy  
            **value**: log value
        """
        ret = None

        assert isinstance(step_logs, dict), "Step logs must be a dictionary."
        for step_name, log in step_logs.items():
            ret_dict = {}
            if self.main_cfg.verbose_flag:
                print(f"Parsing step log: {step_name}")

            if isinstance(log, (str, int, float, list, tuple, bytes)):
                ret_dict[step_name] = log
                if ret is None:
                    ret = [ret_dict,]
                else:
                    ret += [ret_dict,]

            if isinstance(log, dict):
                has_match = any(h in k for k in log for h in self.log_hierarchy)

                if not has_match:
                    _temp = [v for v in log.values() \
                            if isinstance(v, (str, int, float, list, tuple, bytes))]
                    _ret = [{step_name: _temp},] if len(_temp) > 0 else None
                else:
                    _ret = self._dfs_searcher(log, Path(step_name))

                if _ret is not None:
                    ret = _ret if ret is None else ret + _ret

        assert ret is not None, "Failed to parse step logs."
        return tuple(ret)


    def log_parser(self, log_dict: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse Logs data to extract RGB color information.
        """
        if self.main_cfg.verbose_flag:
            print(f"Parsing log dictionary: {log_dict.keys()}")
            print(f"Log dictionary content: {log_dict}")

        if isinstance(log_dict, dict):
            file_name = list(log_dict.keys())
        else:
            raise ValueError("log_dict must be a dictionary.")

        if len(file_name) == 1:
            file_name = file_name[0]
        else:
            raise ValueError("Multiple keys found in log_dict.")
        full_log = log_dict[file_name]

        message = full_log.get("Message", None)
        assert message is not None, "No message found in log."
        step_logs = full_log.get("Logs", None)
        assert step_logs is not None, "No step_logs found in log."

        return file_name, message, step_logs


    def main(self) -> None:
        """
        Main method to convert bytes data to a list of RGB tuples.
        """
        logs = Logs.get_logs(self.io_controller, self.hash_cfg, self.main_cfg, self.log_hierarchy)
        log_process = tqdm(logs, desc="Processing Logs", unit="log")
        for log_dict in log_process:
            self.img_writer(log_dict)
