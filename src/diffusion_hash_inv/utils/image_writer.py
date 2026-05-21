"""
Make RGB images from Logs.
"""
from __future__ import annotations
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Dict, Optional, Sequence

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
from diffusion_hash_inv.utils.ecc48 import SUPPORTED_METHODS
from diffusion_hash_inv.utils.file_io import FileIO
from diffusion_hash_inv.utils.progress import progress
from diffusion_hash_inv.main.context import RuntimeConfig

ImageRecord = Tuple[Path, Image.Image, Any]


def _import_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to write HDF5 tensor datasets. "
            "Install it with `pip install h5py` or `pip install -e '.[hdf5]'`."
        ) from exc
    return h5py


def _is_notebook_kernel() -> bool:
    return "ipykernel" in sys.modules or "JPY_PARENT_PID" in os.environ


def _parallel_executor_type():
    """
    Use threads inside Jupyter to avoid macOS/spawn worker shutdown hangs.
    """
    return ThreadPoolExecutor if _is_notebook_kernel() else ProcessPoolExecutor


def _runtime_config_without_clean(runtime_cfg: RuntimeConfig) -> RuntimeConfig:
    """
    Return a worker-safe runtime config.

    Process workers must never inherit clean_flag=True because each worker creates
    its own FileIO instance. Keeping clean_flag=True there would delete data/output.
    """
    if not runtime_cfg.main.clean_flag:
        return runtime_cfg

    main_cfg = dataclass_replace(runtime_cfg.main, clean_flag=False)
    return dataclass_replace(runtime_cfg, main=main_cfg)


def _chunk_paths(paths: List[Path], chunk_size: int) -> List[List[Path]]:
    return [paths[idx:idx + chunk_size] for idx in range(0, len(paths), chunk_size)]


def _write_log_chunk_worker(
    args: Tuple[RuntimeConfig, Byte2RGBConfig, List[str], List[Path]],
) -> Tuple[int, int]:
    runtime_cfg, rgb_config, log_hierarchy, log_paths = args
    worker_runtime_cfg = _runtime_config_without_clean(runtime_cfg)
    io_controller = FileIO(worker_runtime_cfg.main, worker_runtime_cfg.output)
    image_maker = RGBImgMaker(worker_runtime_cfg, io_controller, rgb_config, quiet=True)
    image_maker.log_hierarchy = list(log_hierarchy)

    logs_processed = 0
    images_written = 0
    for log_dict in Logs.iter_logs_with_hierarchy(
        io_controller,
        image_maker.log_hierarchy,
        list(log_paths),
    ):
        filename, message, parsed_logs = image_maker._parse_image_logs(log_dict)
        images_written += image_maker._write_parsed_images(
            filename,
            message,
            parsed_logs,
        )
        logs_processed += 1

    return logs_processed, images_written


def _write_hdf5_chunk_worker(
    args: Tuple[
        RuntimeConfig,
        Byte2RGBConfig,
        List[str],
        List[Path],
        Path,
        Optional[Tuple[str, ...]],
        int,
        bool,
        Optional[str],
        bool,
    ],
) -> Tuple[Path, int, int]:
    (
        runtime_cfg,
        rgb_config,
        log_hierarchy,
        log_paths,
        shard_path,
        include_paths,
        channels,
        normalize,
        compression,
        preserve_log_hierarchy,
    ) = args
    worker_runtime_cfg = _runtime_config_without_clean(runtime_cfg)
    io_controller = FileIO(worker_runtime_cfg.main, worker_runtime_cfg.output)
    return HDF5Maker._write_hdf5_shard(
        runtime_cfg=worker_runtime_cfg,
        io_controller=io_controller,
        rgb_config=rgb_config,
        log_hierarchy=log_hierarchy,
        log_paths=log_paths,
        shard_path=shard_path,
        include_paths=include_paths,
        channels=channels,
        normalize=normalize,
        compression=compression,
        preserve_log_hierarchy=preserve_log_hierarchy,
    )


class RGBImgMaker:
    """
    A class to make RGB images from Logs.
    """

    def __init__(self, runtime_cfg: RuntimeConfig,
                io_controller: FileIO,
                rgb_config: Byte2RGBConfig,
                quiet: bool = False):
        self.runtime_cfg = runtime_cfg
        self.main_cfg = runtime_cfg.main
        self.hash_cfg = runtime_cfg.hash
        self.io_controller = io_controller
        self.log_hierarchy: Optional[List[str]] = []
        self.byte2rgb = Byte2RGB(main_config=self.main_cfg,
                                hash_config=self.hash_cfg,
                                rgb_config=rgb_config)
        self.log_hierarchy: Optional[List[str]] = []
        if not quiet:
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
            frames.append(Image.fromarray(canvas))

        assert len(frames) > 0, "Failed to create image from RGB data."
        if len(frames) == 1:
            return frames[0]
        return self._image_concater(frames, direction="horizontal")


    def _image_formatter_bg_fg(
        self,
        bg: RGB | RGBA,
        fg: RGB | RGBA,
        image_size: Tuple[int, int],
        center_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Create one image block with *bg* filling the background and *fg* filling the
        center block.  Used by 48-bit encoding modes (``golay24-dual``, ``rs48``,
        ``bch48``) so that the 1st encoded RGB becomes the background and the 2nd
        becomes the foreground.
        """
        assert center_size[0] < image_size[0] or center_size[1] < image_size[1], (
            "center_size must be smaller than image_size so the background is visible. "
            f"Got center_size={center_size}, image_size={image_size}."
        )
        center_x = (image_size[0] - center_size[0]) // 2
        center_y = (image_size[1] - center_size[1]) // 2
        canvas = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
        canvas[:, :] = (bg.r, bg.g, bg.b, 255) if isinstance(bg, RGB) else \
            (bg.r, bg.g, bg.b, bg.a)
        canvas[center_y:center_y + center_size[1], center_x:center_x + center_size[0]] = \
            (fg.r, fg.g, fg.b, 255) if isinstance(fg, RGB) else (fg.r, fg.g, fg.b, fg.a)
        return Image.fromarray(canvas)

    def _image_formatter_48bit(
        self,
        rgb_list: Tuple[RGB | RGBA, ...],
        image_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Render a 48-bit encoded RGB list as composite background+foreground blocks.

        Each consecutive (RGB1, RGB2) pair produces one ``image_size`` block:
        - RGB1 → background fill
        - RGB2 → center foreground block (half of ``image_size`` in each dimension)

        Multiple pairs are concatenated horizontally.
        """
        if len(rgb_list) % 2 != 0:
            raise ValueError(
                f"48-bit encoding requires an even number of RGB values, got {len(rgb_list)}."
            )
        center_size = (image_size[0] // 2, image_size[1] // 2)
        frames: List[Image.Image] = []
        for i in range(0, len(rgb_list), 2):
            frames.append(
                self._image_formatter_bg_fg(rgb_list[i], rgb_list[i + 1], image_size, center_size)
            )
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
        enc = self.byte2rgb.rgb_config.encoding

        if isinstance(rgb_data, Tuple):
            if isinstance(rgb_data[0], (RGB, RGBA)):
                if enc in SUPPORTED_METHODS:
                    return self._image_formatter_48bit(rgb_data, img_size)
                return self._image_formatter(rgb_data, img_size, center_size)

            if isinstance(rgb_data[0], Tuple):
                ret = None
                for data in rgb_data:
                    if not isinstance(data, Tuple):
                        raise ValueError(
                            "All items in rgb_data tuple must be of type Tuple[RGB] or Tuple[RGBA]"
                            )
                    if enc in SUPPORTED_METHODS:
                        img = self._image_formatter_48bit(data, img_size)
                    else:
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
                    raise ValueError("All items in data list must be of type str or bytes."
                                    f" Got item of type {type(item)} with value: {item}")
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

    def _parse_image_logs(
        self,
        log_dict: Dict[str, Any],
    ) -> Tuple[str, str, Tuple[Dict[str, Any], ...]]:
        """
        Parse one log file into the pieces used for image generation.
        """
        filename, message, step_logs = Logs.log_parser(log_dict)
        parsed_logs = Logs.steplogs_parser(step_logs, self.log_hierarchy)

        return filename, message, parsed_logs

    @staticmethod
    def _image_count(parsed_logs: Tuple[Dict[str, Any], ...]) -> int:
        """Return the number of image entries produced for one parsed log."""
        return 1 + len(parsed_logs)  # message image + one image per parsed step log

    @staticmethod
    def _advance_progress(progress_bar: Optional[Any]) -> None:
        """Advance an optional progress bar after one image is written."""
        if progress_bar is not None:
            progress_bar.update(1)

    @staticmethod
    def _normalize_worker_count(workers: Optional[int]) -> int:
        if workers is None:
            return 1
        worker_count = int(workers)
        if worker_count < 1:
            raise ValueError("workers must be greater than or equal to 1")
        return worker_count

    def _format_image_records(
        self,
        message: str,
        parsed_logs: Tuple[Dict[str, Any], ...],
    ) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        try:
            encoded_message = self.data_encoder(message)
            records.append((Path("message.png"), self.image_formatter(encoded_message), message))

            for log in parsed_logs:
                assert isinstance(log, dict), "Parsed log must be a dictionary."
                path = list(log.keys())
                assert len(path) == 1, \
                    f"Parsed log dictionary must have exactly one key. {len(path)} keys found."
                path = path[0]
                data = log[path]
                file_name = path.split("/")[-1]
                assert isinstance(data, (str, int, float, list, tuple, bytes)), \
                    "Parsed log data must be of type str, int, float, list, tuple, or bytes."
                encoded_log = self.data_encoder(data)
                if self.main_cfg.verbose_flag:
                    print(encoded_log)

                parent_path = Path("/".join(path.split("/")[:-1]))
                records.append((
                    parent_path / f"{file_name}.png",
                    self.image_formatter(encoded_log),
                    data,
                ))
        except Exception:
            for _, image, _ in records:
                image.close()
            raise

        return records

    def _png_output_path(self, filename: str, relative_path: Path) -> Path:
        output_relative_path = Path(filename, relative_path.parent, relative_path.name)
        return (
            self.io_controller.data_dir
            / "images"
            / self.io_controller._sanitize_relative_path(output_relative_path)
        )

    def _sample_image_rgb_rows(self, image: Image.Image) -> List[Tuple[RGB, ...]]:
        """
        Sample RGB values from a saved PNG image, one row per ``ImgConfig.img_size``
        block row.

        For 48-bit encoding modes (``golay24-dual``, ``rs48``, ``bch48``), each block
        encodes one byte as a background+foreground pair.  Two pixels are sampled per
        block in interleaved order — ``(bg_pixel, fg_pixel)`` — so that the flat pixel
        list feeds directly into :meth:`~Byte2RGB.rgb_decoder`.

        For all other encodings, the center pixel of each block is sampled (legacy
        behaviour).
        """
        enc = self.byte2rgb.rgb_config.encoding
        block_width, block_height = ImgConfig().img_size
        width, height = image.size
        if width % block_width != 0 or height % block_height != 0:
            raise ValueError(
                "Saved image dimensions must be multiples of ImgConfig.img_size. "
                f"Got {(width, height)}, block size {(block_width, block_height)}."
            )

        rgb_image = image.convert("RGB")
        try:
            rows: List[Tuple[RGB, ...]] = []
            if enc in SUPPORTED_METHODS:
                # bg+fg layout: sample corner (background = RGB1) and center (foreground = RGB2)
                # Background pixel is at offset (1, 1) from block top-left, which is always
                # within the border area (center starts at block_width//4, block_height//4).
                bg_x_off = 1
                bg_y_off = 1
                fg_x_off = block_width // 2
                fg_y_off = block_height // 2
                for y_idx in range(height // block_height):
                    row: List[RGB] = []
                    y_bg = y_idx * block_height + bg_y_off
                    y_fg = y_idx * block_height + fg_y_off
                    for x_idx in range(width // block_width):
                        x_bg = x_idx * block_width + bg_x_off
                        x_fg = x_idx * block_width + fg_x_off
                        row.append(RGB.from_tuple(rgb_image.getpixel((x_bg, y_bg))))  # RGB1 (bg)
                        row.append(RGB.from_tuple(rgb_image.getpixel((x_fg, y_fg))))  # RGB2 (fg)
                    rows.append(tuple(row))
            else:
                for y_idx in range(height // block_height):
                    row = []
                    y = y_idx * block_height + block_height // 2
                    for x_idx in range(width // block_width):
                        x = x_idx * block_width + block_width // 2
                        row.append(RGB.from_tuple(rgb_image.getpixel((x, y))))
                    rows.append(tuple(row))
            return rows
        finally:
            rgb_image.close()

    def _validate_saved_png(self, path: Path, original_data: Any) -> None:
        with Image.open(path) as image:
            rows = self._sample_image_rgb_rows(image)

        if isinstance(original_data, list):
            if not all(isinstance(item, (str, bytes)) for item in original_data):
                bad_type = next(type(item) for item in original_data
                                if not isinstance(item, (str, bytes)))
                raise ValueError(
                    "Saved PNG validation only supports str/bytes list items. "
                    f"Got {bad_type} for {path}."
                )
            if len(rows) != len(original_data):
                decoded_rgb = tuple(pixel for row in rows for pixel in row)
                expected = b"".join(
                    Logs.str_to_bytes(item) if isinstance(item, str) else item
                    for item in original_data
                )
                if encoding_validate(expected, decoded_rgb, self.byte2rgb):
                    return
                raise RuntimeError(
                    f"Saved PNG validation failed for {path}: "
                    f"decoded {len(rows)} rows, expected {len(original_data)} rows."
                )
            for row_idx, (row, item) in enumerate(zip(rows, original_data), start=1):
                if not encoding_validate(item, row, self.byte2rgb):
                    raise RuntimeError(
                        f"Saved PNG validation failed for {path} at row {row_idx}."
                    )
            return

        if not isinstance(original_data, (str, bytes)):
            raise ValueError(
                "Saved PNG validation only supports str, bytes, or list[str|bytes]. "
                f"Got {type(original_data)} for {path}."
            )

        decoded_rgb = tuple(pixel for row in rows for pixel in row)
        if not encoding_validate(original_data, decoded_rgb, self.byte2rgb):
            raise RuntimeError(f"Saved PNG validation failed for {path}.")

    def _write_png_records(
        self,
        filename: str,
        records: List[ImageRecord],
        image_process: Optional[Any] = None,
    ) -> int:
        for relative_path, image, original_data in records:
            self.io_controller.file_writer(
                relative_path.name,
                image,
                parent_dir=Path(filename, relative_path.parent),
                data_type="data",
            )
            self._validate_saved_png(
                self._png_output_path(filename, relative_path),
                original_data,
            )
            self._advance_progress(image_process)
        return len(records)

    def _write_parsed_images(
        self,
        filename: str,
        message: str,
        parsed_logs: Tuple[Dict[str, Any], ...],
        image_process: Optional[Any] = None,
    ) -> int:
        """
        Write already-parsed log data as PNG images.
        """
        records = self._format_image_records(message, parsed_logs)
        try:
            return self._write_png_records(filename, records, image_process)
        finally:
            for _, image, _ in records:
                image.close()

    def img_writer(self, log_dict: Dict[str, Any],
                image_process: Optional[Any] = None) -> int:
        """
        Write RGB image data to file.
        """
        filename, message, parsed_logs = self._parse_image_logs(log_dict)
        return self._write_parsed_images(
            filename,
            message,
            parsed_logs,
            image_process,
        )

    def _write_parallel(
        self,
        selected_logs: List[Path],
        workers: int,
        parallel_chunk_size: Optional[int],
    ) -> int:
        chunk_size = parallel_chunk_size
        if chunk_size is None:
            chunk_size = max(1, min(256, math.ceil(len(selected_logs) / (workers * 8))))
        chunk_size = max(1, int(chunk_size))
        chunks = _chunk_paths(selected_logs, chunk_size)
        executor_type = _parallel_executor_type()
        print(
            f"Parallel image writing: {workers} workers, {len(chunks)} chunks, "
            f"chunk_size={chunk_size}, executor={executor_type.__name__}."
        )

        images_written = 0
        logs_processed = 0
        futures = []
        log_process = progress((), total=len(selected_logs), desc="Processing Logs", unit="log")

        with executor_type(max_workers=workers) as executor, log_process:
            for chunk in chunks:
                futures.append(executor.submit(
                    _write_log_chunk_worker,
                    (
                        self.runtime_cfg,
                        self.byte2rgb.rgb_config,
                        list(self.log_hierarchy or []),
                        chunk,
                    ),
                ))

            for future in as_completed(futures):
                chunk_logs, chunk_images = future.result()
                logs_processed += chunk_logs
                images_written += chunk_images
                log_process.update(chunk_logs)
                log_process.set_postfix({"images": images_written})

        assert logs_processed == len(selected_logs), \
            f"Processed {logs_processed} logs, expected {len(selected_logs)}"
        return images_written

    def main(
        self,
        logs: Optional[Iterable[Path | str]] = None,
        *,
        workers: Optional[int] = None,
        parallel_chunk_size: Optional[int] = None,
    ) -> int:
        """
        Main method to convert logs to PNG images.
        """
        if workers is None:
            workers = getattr(self.main_cfg, "image_workers", 1)
        worker_count = self._normalize_worker_count(workers)

        if logs is None:
            selected_logs = self.io_controller.get_latest_files_by_date(
                self.hash_cfg.hash_alg,
                self.hash_cfg.length,
            )
        else:
            selected_logs = sorted(Path(log) for log in logs)

        print(f"Found {len(selected_logs)} logs to process.")
        assert len(selected_logs) > 0, "No Logs files found."

        first_log = next(
            Logs.iter_logs_with_hierarchy(
                self.io_controller,
                self.log_hierarchy,
                [selected_logs[0]],
            )
        )
        _, _, first_parsed_logs = self._parse_image_logs(first_log)
        images_per_log = self._image_count(first_parsed_logs)
        expected_image_total = images_per_log * len(selected_logs)
        print(f"Expected {expected_image_total} PNG files "
            f"({images_per_log} images per log).")

        if worker_count > 1 and len(selected_logs) > 1:
            worker_count = min(worker_count, len(selected_logs))
            images_written = self._write_parallel(
                selected_logs,
                worker_count,
                parallel_chunk_size,
            )
            print(f"Image writing completed: {images_written} images.")
            return images_written

        log_process = progress((), total=len(selected_logs), desc="Processing Logs", unit="log")
        image_process = progress(
            (),
            total=expected_image_total,
            desc="Writing Images",
            unit="image",
            mininterval=5.0,
        )
        images_written = 0

        with log_process, image_process:
            for log_dict in Logs.iter_logs_with_hierarchy(
                self.io_controller,
                self.log_hierarchy,
                selected_logs,
            ):
                filename, message, parsed_logs = self._parse_image_logs(log_dict)
                image_count = self._image_count(parsed_logs)
                if image_count != images_per_log:
                    image_process.total += image_count - images_per_log

                images_written += self._write_parsed_images(
                    filename,
                    message,
                    parsed_logs,
                    image_process,
                )
                log_process.update(1)
                log_process.set_postfix({"images": images_written})
        print(f"Image writing completed: {images_written} images.")
        return images_written

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
                io_controller: FileIO,
                quiet: bool = False):
        self.runtime_cfg = runtime_cfg
        self.main_cfg = runtime_cfg.main
        self.hash_cfg = runtime_cfg.hash
        self.io_controller = io_controller
        self.log_hierarchy: Optional[List[str]] = []

        if not quiet:
            print("HDF5 Maker Initialized.")

    @staticmethod
    def _normalize_channels(channels: int) -> int:
        if channels not in (1, 3, 4):
            raise ValueError("channels must be 1, 3, or 4")
        return channels

    @staticmethod
    def _pil_to_tensor_array(
        image: Image.Image,
        *,
        channels: int = 3,
        normalize: bool = True,
    ) -> np.ndarray:
        channels = HDF5Maker._normalize_channels(channels)
        if channels == 1:
            array = np.asarray(image.convert("L"), dtype=np.uint8)[None, :, :].copy()
        else:
            mode = "RGB" if channels == 3 else "RGBA"
            array = np.asarray(image.convert(mode), dtype=np.uint8).transpose(2, 0, 1).copy()

        if not normalize:
            return array
        return (array.astype(np.float32) / 127.5) - 1.0

    @staticmethod
    def _hierarchy_segments(relative_path: Path) -> Tuple[str, ...]:
        if relative_path == Path("message.png"):
            return ("Message", "Hex")

        parts = list(relative_path.parts)
        if not parts:
            raise ValueError("relative_path must not be empty")
        parts[-1] = Path(parts[-1]).stem
        return ("Logs", *parts)

    @staticmethod
    def _link_hierarchical_tensor(
        *,
        h5_file: Any,
        source_log: str,
        relative_path: Path,
        tensor_dataset: Any,
        record_key: str,
    ) -> None:
        source_group = h5_file.require_group("logs").require_group(source_log)
        leaf_group = source_group
        for segment in HDF5Maker._hierarchy_segments(relative_path):
            leaf_group = leaf_group.require_group(segment)

        if "tensor" in leaf_group:
            del leaf_group["tensor"]
        leaf_group["tensor"] = tensor_dataset
        leaf_group.attrs["record_key"] = record_key
        leaf_group.attrs["path"] = str(relative_path)
        leaf_group.attrs["source_log"] = source_log

    @staticmethod
    def _write_hdf5_shard(
        *,
        runtime_cfg: RuntimeConfig,
        io_controller: FileIO,
        rgb_config: Byte2RGBConfig,
        log_hierarchy: Sequence[str],
        log_paths: Sequence[Path],
        shard_path: Path,
        include_paths: Optional[Tuple[str, ...]],
        channels: int,
        normalize: bool,
        compression: Optional[str],
        preserve_log_hierarchy: bool = True,
    ) -> Tuple[Path, int, int]:
        h5py = _import_h5py()
        channels = HDF5Maker._normalize_channels(channels)
        include_path_set = set(include_paths) if include_paths is not None else None

        image_maker = RGBImgMaker(runtime_cfg, io_controller, rgb_config, quiet=True)
        image_maker.log_hierarchy = list(log_hierarchy)

        shard_path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        source_logs: List[str] = []
        record_paths: List[str] = []
        record_keys: List[str] = []
        logs_processed = 0
        tensors_written = 0

        with h5py.File(shard_path, "w") as h5_file:
            h5_file.attrs["format"] = "diffusion_hash_inv.hdf5_tensor_shard"
            h5_file.attrs["channels"] = channels
            h5_file.attrs["normalized"] = normalize
            h5_file.attrs["normalization"] = "[-1, 1]" if normalize else "uint8 [0, 255]"
            h5_file.attrs["tensor_layout"] = "C,H,W"
            h5_file.attrs["compression"] = "" if compression is None else compression
            h5_file.attrs["preserve_log_hierarchy"] = preserve_log_hierarchy
            records_group = h5_file.create_group("records")
            if preserve_log_hierarchy:
                h5_file.create_group("logs")

            for log_dict in Logs.iter_logs_with_hierarchy(
                io_controller,
                image_maker.log_hierarchy,
                list(log_paths),
            ):
                filename, message, parsed_logs = image_maker._parse_image_logs(log_dict)
                records = image_maker._format_image_records(message, parsed_logs)
                try:
                    for relative_path, image, _ in records:
                        path_text = str(relative_path)
                        if include_path_set is not None and path_text not in include_path_set:
                            continue

                        tensor = HDF5Maker._pil_to_tensor_array(
                            image,
                            channels=channels,
                            normalize=normalize,
                        )
                        record_key = f"{tensors_written:08d}"
                        record_group = records_group.create_group(record_key)
                        tensor_dataset = record_group.create_dataset(
                            "tensor",
                            data=tensor,
                            compression=compression,
                            shuffle=compression is not None,
                        )
                        record_group.attrs["source_log"] = filename
                        record_group.attrs["path"] = path_text
                        record_group.attrs["shape"] = tensor.shape
                        if preserve_log_hierarchy:
                            HDF5Maker._link_hierarchical_tensor(
                                h5_file=h5_file,
                                source_log=filename,
                                relative_path=relative_path,
                                tensor_dataset=tensor_dataset,
                                record_key=record_key,
                            )
                        source_logs.append(filename)
                        record_paths.append(path_text)
                        record_keys.append(record_key)
                        tensors_written += 1
                finally:
                    for _, image, _ in records:
                        image.close()
                logs_processed += 1

            h5_file.create_dataset("source_logs", data=np.array(source_logs, dtype=object),
                                dtype=string_dtype)
            h5_file.create_dataset("paths", data=np.array(record_paths, dtype=object),
                                dtype=string_dtype)
            h5_file.create_dataset("record_keys", data=np.array(record_keys, dtype=object),
                                dtype=string_dtype)
            h5_file.attrs["log_count"] = logs_processed
            h5_file.attrs["tensor_count"] = tensors_written

        return shard_path, logs_processed, tensors_written

    def main(
        self,
        logs: Optional[Iterable[Path | str]] = None,
        *,
        rgb_config: Optional[Byte2RGBConfig] = None,
        workers: Optional[int] = None,
        output_dir: Optional[Path | str] = None,
        output_name: str = "hash_tensors",
        shard_size: int = 256,
        include_paths: Optional[Sequence[str]] = None,
        channels: int = 3,
        normalize: bool = True,
        compression: Optional[str] = "gzip",
        preserve_log_hierarchy: bool = True,
    ) -> List[Path]:
        """
        Build sharded HDF5 tensor datasets from JSON logs.

        Each process writes an independent HDF5 shard. This avoids unsafe
        concurrent writes to a single HDF5 file while still parallelizing the
        CPU-heavy log-to-tensor conversion.

        When ``preserve_log_hierarchy`` is true, the same tensor datasets are
        also hard-linked below ``logs/<source_log>/Message/Hex`` and
        ``logs/<source_log>/Logs/...`` so the HDF5 file mirrors the JSON log
        hierarchy without duplicating tensor storage.
        """
        if rgb_config is None:
            rgb_config = self.runtime_cfg.rgb
        if workers is None:
            workers = getattr(self.main_cfg, "image_workers", 1)
        worker_count = RGBImgMaker._normalize_worker_count(workers)
        channels = self._normalize_channels(channels)
        if shard_size < 1:
            raise ValueError("shard_size must be greater than or equal to 1")

        if logs is None:
            selected_logs = self.io_controller.get_latest_files_by_date(
                self.hash_cfg.hash_alg,
                self.hash_cfg.length,
            )
        else:
            selected_logs = sorted(Path(log) for log in logs)

        print(f"Found {len(selected_logs)} logs for HDF5 tensor dataset.")
        assert len(selected_logs) > 0, "No Logs files found."

        dataset_dir = (
            Path(output_dir)
            if output_dir is not None
            else self.io_controller.data_dir / "tensor_datasets" / output_name
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)

        chunks = _chunk_paths(selected_logs, shard_size)
        include_paths_tuple = tuple(include_paths) if include_paths is not None else None
        worker_count = min(worker_count, len(chunks))
        print(
            f"HDF5 tensor dataset: {len(chunks)} shards, {worker_count} workers, "
            f"shard_size={shard_size}."
        )

        results: List[Tuple[Path, int, int]] = []
        log_process = progress((), total=len(selected_logs), desc="Writing HDF5", unit="log")
        with log_process:
            if worker_count > 1:
                executor_type = _parallel_executor_type()
                print(f"HDF5 executor: {executor_type.__name__}.")
                with executor_type(max_workers=worker_count) as executor:
                    futures = []
                    for shard_idx, chunk in enumerate(chunks):
                        shard_path = dataset_dir / f"{output_name}_{shard_idx:06d}.h5"
                        futures.append(executor.submit(
                            _write_hdf5_chunk_worker,
                            (
                                self.runtime_cfg,
                                rgb_config,
                                list(self.log_hierarchy or []),
                                chunk,
                                shard_path,
                                include_paths_tuple,
                                channels,
                                normalize,
                                compression,
                                preserve_log_hierarchy,
                            ),
                        ))

                    for future in as_completed(futures):
                        shard_path, log_count, tensor_count = future.result()
                        results.append((shard_path, log_count, tensor_count))
                        log_process.update(log_count)
                        log_process.set_postfix({"tensors": sum(result[2] for result in results)})
            else:
                for shard_idx, chunk in enumerate(chunks):
                    shard_path = dataset_dir / f"{output_name}_{shard_idx:06d}.h5"
                    result = self._write_hdf5_shard(
                        runtime_cfg=_runtime_config_without_clean(self.runtime_cfg),
                        io_controller=self.io_controller,
                        rgb_config=rgb_config,
                        log_hierarchy=list(self.log_hierarchy or []),
                        log_paths=chunk,
                        shard_path=shard_path,
                        include_paths=include_paths_tuple,
                        channels=channels,
                        normalize=normalize,
                        compression=compression,
                        preserve_log_hierarchy=preserve_log_hierarchy,
                    )
                    results.append(result)
                    log_process.update(result[1])
                    log_process.set_postfix({"tensors": sum(item[2] for item in results)})

        results.sort(key=lambda item: item[0].name)
        manifest = {
            "format": "diffusion_hash_inv.hdf5_tensor_manifest",
            "output_name": output_name,
            "log_count": sum(item[1] for item in results),
            "tensor_count": sum(item[2] for item in results),
            "shard_count": len(results),
            "channels": channels,
            "normalized": normalize,
            "include_paths": list(include_paths_tuple) if include_paths_tuple is not None else None,
            "preserve_log_hierarchy": preserve_log_hierarchy,
            "shards": [
                {
                    "file": item[0].name,
                    "log_count": item[1],
                    "tensor_count": item[2],
                }
                for item in results
            ],
        }
        (dataset_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding=self.io_controller.encoding,
        )
        print(
            f"HDF5 tensor dataset completed: {manifest['tensor_count']} tensors "
            f"in {manifest['shard_count']} shards."
        )
        return [item[0] for item in results]


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
