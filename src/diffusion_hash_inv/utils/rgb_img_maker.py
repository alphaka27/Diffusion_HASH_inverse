"""
Make RGB images from Logs.
"""
from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np

from diffusion_hash_inv.config import MainConfig, HashConfig, ImgConfig
from diffusion_hash_inv.core import RGB, RGBA
from diffusion_hash_inv.utils import Byte2RGB
from diffusion_hash_inv.utils import FileIO
from diffusion_hash_inv.validation import encoding_validate

class RGBImgMaker:
    """
    A class to make RGB images from Logs.
    """

    def __init__(self, main_cfg: MainConfig, hash_cfg: HashConfig, io_controller: FileIO):
        self.main_cfg = main_cfg
        self.hash_cfg = hash_cfg
        self.io_controller = io_controller
        self.byte2rgb = Byte2RGB(main_config=self.main_cfg, hash_config=self.hash_cfg)
        self.log_hierarchy: Dict[str, Any] = {}

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

        canvas = np.empty((image_size[1], image_size[0], 4), dtype=np.uint8)
        background_color = (255, 255, 255, 255)  # White with full opacity
        canvas[:, :] = background_color
        center_x = (image_size[0] - center_size[0]) // 2
        center_y = (image_size[1] - center_size[1]) // 2
        assert center_x >= 0 and center_y >= 0, "Center size must be smaller than image size."

        assert len(rgb_data) > 0, "RGB data cannot be empty."
        assert all(isinstance(pixel, (RGB, RGBA)) for pixel in rgb_data), \
            "All items in rgb_data must be of type RGB or RGBA."
        assert image_size[0] > center_size[0] + center_x and \
            image_size[1] > center_size[1] + center_y, \
            "Image size must be large enough to accommodate center size with offset."

        ret = None
        for rgb in rgb_data:
            canvas[center_y:center_y + center_size[1], center_x:center_x + center_size[0]] = \
                (rgb.r, rgb.g, rgb.b, 255) if isinstance(rgb, RGB) else (rgb.r, rgb.g, rgb.b, rgb.a)
            if ret is None:
                ret = Image.fromarray(canvas, "RGBA")
            else:
                ret = self._image_concater(
                    [ret,
                    Image.fromarray(canvas, "RGBA")],
                    direction="horizontal")
        assert ret is not None, "Failed to create image from RGB data."
        return ret

    def image_formatter(self, \
                        rgb_data: Tuple[RGB] | List[Tuple[RGB]] | Tuple[RGBA] | List[Tuple[RGBA]]) \
                        -> Image.Image:
        """
        Make RGB image from RGB data.
        """
        img_size = (ImgConfig().img_size[0], ImgConfig().img_size[1])  # Width, Height
        center_size = (ImgConfig().center_size[0], ImgConfig().center_size[1])  # Width, Height

        if isinstance(rgb_data, Tuple):
            return self._image_formatter(rgb_data, img_size, center_size)

        if isinstance(rgb_data, List):
            ret = None
            for data in rgb_data:
                if not isinstance(data, Tuple):
                    raise ValueError("All items in rgb_data list must be of type Tuple[RGB]")
                ret = self._image_formatter(data, img_size, center_size) if ret is None else \
                    self._image_concater(
                        [ret,
                        self._image_formatter(data, img_size, center_size)],
                        direction="vertical")
            return ret

        raise ValueError("rgb_data must be of type Tuple[RGB] or List[Tuple[RGB]]")


    def get_logs(self) -> List[Dict]:
        """
        Get Logs data from file.
        """
        latest_logs = \
            self.io_controller.get_latest_files_by_date(self.hash_cfg.hash_alg, \
                                                        self.hash_cfg.length)
        latest_logs.sort()
        logs: List[Dict] = []
        hierarchy: Dict[str, Any] = None

        assert len(latest_logs) > 0, "No Logs files found."
        if self.main_cfg.verbose_flag:
            print(f"Found {len(latest_logs)} Logs files.")

        for log_file in latest_logs:
            if self.main_cfg.verbose_flag:
                print(f"Loading Logs from file: {log_file}")
            log = self.io_controller.file_reader(log_file, length=self.hash_cfg.length)
            _hierarchy = log.get("Hierarchy", None)
            assert _hierarchy is not None, "No Hierarchy found in Logs."
            if hierarchy is None:
                hierarchy = _hierarchy
            else:
                assert hierarchy == _hierarchy, "Hierarchy mismatch in Logs."
            logs.append({log_file.stem: log})

        self.log_hierarchy = hierarchy
        return logs

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

        raise ValueError("Unsupported data type for encoding.")

    def img_writer(self, logs: List[Dict]) -> None:
        """
        Write RGB image data to file.
        """
        for log_dict in logs:
            filename, message, step_logs = self.log_parser(log_dict)

            parsed_logs = self.steplogs_parser(step_logs)
            if self.main_cfg.verbose_flag:
                print(f"Parsed step logs: {parsed_logs}")

            encoded_message = self.data_encoder(message)
            print(type(encoded_message), len(encoded_message))
            rgb_message = self.image_formatter(encoded_message)

            self.io_controller.file_writer("message.png",
                                        rgb_message,
                                        parent_dir=filename,
                                        data_type="data")
            for log in parsed_logs:
                assert isinstance(log, dict), "Parsed log must be a dictionary."
                path = list(log.keys())
                assert len(path) == 1, "Parsed log dictionary must have exactly one key."
                path = path[0]
                data = log[path]
                assert isinstance(data, (str, int, float, list, tuple, bytes)), \
                    "Parsed log data must be of type str, int, float, list, tuple, or bytes."

                encoded_log = self.data_encoder(data)
                rgb_log = self.image_formatter(encoded_log)
                self.io_controller.file_writer(f"{data}.png",
                                            rgb_log,
                                            parent_dir=Path(filename, path),
                                            data_type="data")


    def _dfs_searcher(self, data_dict: Dict[str, Any], key_path: Path = None) \
        -> Tuple[Dict[str, Any]]:
        """
        Depth-first search to traverse the log hierarchy.
        """
        _key_path = key_path if key_path is not None else None

        for key, value in data_dict.items():
            if self.main_cfg.verbose_flag:
                print(f"Visiting node: {key}")

            if isinstance(value, dict):
                _key_path = key_path / key if key_path is not None else Path(key)
                if self.main_cfg.verbose_flag:
                    print(f"Descending into dictionary at node: {key} with path: {_key_path}")
                self._dfs_searcher(value, _key_path)
            else:
                if self.main_cfg.verbose_flag:
                    print(f"Reached leaf node: {key} with value: {value}")

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
                pass


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
        logs = self.get_logs()
        self.img_writer(logs)
