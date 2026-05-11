"""
Conditional DDPM training pipeline for generated hash images.

The generated images are expected to live under a structure like:

    data/images/<run-id>/message.png
    data/images/<run-id>/4th Step/1st Round/57th Loop.png

Only ``message.png`` files are used for training. The first directory under
``data/images`` is treated as the run id and matched to
``output/json/**/<run-id>.json``. The condition label is the final hash value
from the matching run JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_hash_inv.analyze import Analyze
from diffusion_hash_inv.config import ImgConfig
from diffusion_hash_inv.scheduling import BetaScheduler

ConditionMode = Literal[
    "json-step",
    "json-value",
    "json-step-value",
    "relative-path",
    "top-level",
    "filename",
]
FitMode = Literal["pad", "resize", "reshape", "height-flatten"]
BetaScheduleMode = Literal["linear", "file", "hash-approach1", "hash-approach2"]
LabelSource = Literal["final-hash"]

# Temporal conditioning mode: how the loop-position signal is encoded.
#   "class"           – original discrete nn.Embedding (no temporal ordering).
#   "loop-sinusoidal" – Method 1: sinusoidal PE on loop_idx (adjacent loops ≈ similar).
#   "loop-structured" – Method 2: sinusoidal(loop_idx) + MLP(loop_start) + MLP(loop_end).
#   "loop-sequence"   – Method 3: Transformer over the full 64-loop token sequence;
#                       each position is contextualised by all others.
TemporalConditioningMode = Literal[
    "class",
    "loop-sinusoidal",
    "loop-structured",
    "loop-sequence",
]


@dataclass(frozen=True)
class GeneratedImageSample:
    """One generated image and the condition label derived from JSON metadata."""

    path: Path
    condition: str
    label: int
    # Temporal loop metadata (populated when use_loop_images=True or from path).
    # Defaults represent "no loop context" (single message.png per run).
    loop_idx: int = 0          # 0-based index within the loop sequence
    loop_count: int = 1        # total number of loops in the sequence
    loop_start: float = 0.0   # normalised start position = loop_idx / loop_count
    loop_end: float = 1.0     # normalised end position = (loop_idx+1) / loop_count


@dataclass(frozen=True)
class ConditionalDiffusionTrainConfig:
    """Runtime configuration for training on generated images."""

    data_root: Path = Path("data/images")
    json_root: Path = Path("output/json")
    output_dir: Path = Path("output/conditional_diffusion")
    image_size: int = 64
    channels: int = 3
    fit_mode: FitMode = "reshape"
    condition_mode: ConditionMode = "json-step"
    label_source: LabelSource = "final-hash"
    max_images: int | None = None
    batch_size: int = 32
    train_steps: int = 1_000
    epochs: int | None = None
    timesteps: int | Literal["auto"] = 1_000
    learning_rate: float = 2e-4
    base_channels: int = 64
    time_dim: int = 256
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: BetaScheduleMode = "linear"
    beta_values_path: Path | None = None
    beta_schedule_step: str = "4th Step"
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0
    log_every: int = 50
    sample_every: int = 0
    checkpoint_every: int = 0
    sample_count: int = 16
    save_process_traces: bool = False
    trace_sample_count: int = 4
    trace_steps: int = 8
    save_train_batches_every: int = 0
    # ── Temporal conditioning ─────────────────────────────────────────────────
    # temporal_conditioning selects the loop-position encoding strategy:
    #   "class"           – discrete nn.Embedding (original, no ordering).
    #   "loop-sinusoidal" – Method 1: sinusoidal PE on loop_idx.
    #   "loop-structured" – Method 2: sinusoidal(idx) + MLP(start) + MLP(end).
    #   "loop-sequence"   – Method 3: Transformer over the full loop sequence.
    temporal_conditioning: TemporalConditioningMode = "class"
    # When True, discover 'NNth Loop.png' images instead of message.png.
    use_loop_images: bool = False
    # Maximum number of loop positions the sequence conditioner is built for.
    max_loop_count: int = 64


def _condition_from_relative_path(relative_path: Path, mode: ConditionMode) -> str:
    path_without_suffix = relative_path.with_suffix("")
    if mode in ("json-step", "json-value", "json-step-value"):
        raise ValueError(f"{mode} labels must be read from output JSON files")
    if mode == "relative-path":
        return path_without_suffix.as_posix()
    if mode == "top-level":
        return path_without_suffix.parts[0]
    if mode == "filename":
        return path_without_suffix.name
    raise ValueError(f"Unsupported condition mode: {mode}")


def _load_json_index(json_root: Path | str) -> dict[str, Path]:
    root = Path(json_root)
    if not root.exists():
        raise FileNotFoundError(f"JSON root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"JSON root must be a directory: {root}")

    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            index.setdefault(path.stem, path)
    return index


def _canonical_json_label(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_json_object_label(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json_payload(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"JSON payload file is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON payload in: {path} ({exc.msg} at line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in: {path}")
    return payload


def _lookup_nested_value(payload: dict[str, object], parts: tuple[str, ...]) -> object:
    cursor: object = payload
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            joined = "/".join(parts)
            raise KeyError(f"JSON label path not found: {joined}")
        cursor = cursor[part]
    return cursor


def _json_step_from_image_path(relative_path: Path) -> tuple[str, tuple[str, ...]]:
    if len(relative_path.parts) < 2:
        raise ValueError(
            "JSON labels require images to be stored under data/images/<run-id>/..."
        )

    condition_relative = Path(*relative_path.parts[1:]).with_suffix("")
    if condition_relative.as_posix() == "message":
        return "Message.Hex", ("Message", "Hex")
    return (
        "Logs/" + condition_relative.as_posix(),
        ("Logs", *condition_relative.parts),
    )


def _json_label_from_image_path(
    relative_path: Path,
    payload: dict[str, object],
    mode: ConditionMode,
) -> str:
    step_path, json_parts = _json_step_from_image_path(relative_path)
    value = _lookup_nested_value(payload, json_parts)

    if mode == "json-step":
        return step_path
    if mode == "json-value":
        return _canonical_json_label(value)
    if mode == "json-step-value":
        return _canonical_json_object_label({"step": step_path, "value": value})
    raise ValueError(f"Unsupported JSON condition mode: {mode}")


def _uses_json_condition(mode: ConditionMode) -> bool:
    return mode in ("json-step", "json-value", "json-step-value")


def _final_hash_label_from_payload(payload: dict[str, object]) -> str:
    for key in ("Generated hash", "Correct   hash", "Correct hash"):
        value = payload.get(key)
        if value is not None:
            return _canonical_json_label(value)
    raise KeyError("JSON label path not found: Generated hash")


def _label_from_payload(payload: dict[str, object], label_source: LabelSource) -> str:
    if label_source == "final-hash":
        return _final_hash_label_from_payload(payload)
    raise ValueError(f"Unsupported label source: {label_source}")


# ──────────────────────────────────────────────────────────────────────────────
# Loop temporal metadata helpers
# ──────────────────────────────────────────────────────────────────────────────

# Matches "57th Loop", "1st Loop", "2nd loop", etc.
_LOOP_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+[Ll]oop", re.IGNORECASE)


def _parse_loop_ordinal(name: str) -> int | None:
    """Return 0-based loop index from a name like '57th Loop', or None."""
    match = _LOOP_ORDINAL_RE.search(name)
    if match is None:
        return None
    return int(match.group(1)) - 1  # convert 1-based ordinal to 0-based index


def _loop_normalised_bounds(loop_idx: int, loop_count: int) -> tuple[float, float]:
    """Return (loop_start, loop_end) as normalised positions in [0, 1]."""
    denom = max(loop_count, 1)
    return loop_idx / denom, (loop_idx + 1) / denom


def discover_generated_image_samples(
    root: Path | str,
    condition_mode: ConditionMode = "json-step",
    json_root: Path | str = Path("output/json"),
    label_source: LabelSource = "final-hash",
    max_images: int | None = None,
) -> tuple[list[GeneratedImageSample], dict[str, int]]:
    """
    Discover ``message.png`` images and assign stable integer condition labels.

    The label map is sorted by condition name so that repeated runs over the
    same directory produce the same condition ids. Labels are final hash values
    from the matching run JSON.
    """

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Generated image root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Generated image root must be a directory: {root}")

    image_paths = sorted(path for path in root.rglob("message.png") if path.is_file())
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive when provided")
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No message.png images found under: {root}")

    # `condition_mode` is kept for CLI/config backward compatibility.
    json_index = _load_json_index(json_root)
    payload_cache: dict[str, dict[str, object]] = {}
    condition_names: list[str] = []
    unlabeled: list[tuple[Path, str]] = []
    for path in image_paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(
                "message.png images must be stored under data/images/<run-id>/message.png"
            )
        run_id = relative.parts[0]
        if run_id not in json_index:
            raise FileNotFoundError(f"No JSON file found for image run: {run_id}")
        if run_id not in payload_cache:
            payload_cache[run_id] = _read_json_payload(json_index[run_id])
        condition = _label_from_payload(payload_cache[run_id], label_source)
        condition_names.append(condition)
        unlabeled.append((path, condition))

    condition_to_idx = {name: idx for idx, name in enumerate(sorted(set(condition_names)))}
    samples = [
        GeneratedImageSample(path=path, condition=condition, label=condition_to_idx[condition])
        for path, condition in unlabeled
    ]
    return samples, condition_to_idx


def discover_loop_image_samples(
    root: Path | str,
    json_root: Path | str = Path("output/json"),
    label_source: LabelSource = "final-hash",
    max_images: int | None = None,
    max_loop_count: int = 64,
) -> tuple[list[GeneratedImageSample], dict[str, int]]:
    """
    Discover per-loop images (``NNth Loop.png``) and assign temporal metadata.

    Unlike :func:`discover_generated_image_samples` (which only collects
    ``message.png``), this function collects images whose filename matches the
    pattern ``NNth Loop.png`` (e.g. ``57th Loop.png``).  Each image is tagged
    with:

    * ``loop_idx``   – 0-based position within the 64-step sequence.
    * ``loop_count`` – total loop count found for that run (capped at
                       ``max_loop_count``).
    * ``loop_start`` – ``loop_idx / loop_count`` (normalised position).
    * ``loop_end``   – ``(loop_idx + 1) / loop_count``.

    The condition label is read from JSON in the same way as
    :func:`discover_generated_image_samples`.
    """

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Generated image root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Generated image root must be a directory: {root}")

    # Collect only files that match the loop pattern
    all_loop_paths: list[Path] = sorted(
        path
        for path in root.rglob("*.png")
        if path.is_file() and _parse_loop_ordinal(path.stem) is not None
    )
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive when provided")
        all_loop_paths = all_loop_paths[:max_images]
    if not all_loop_paths:
        raise ValueError(f"No loop images found under: {root}")

    json_index = _load_json_index(json_root)
    payload_cache: dict[str, dict[str, object]] = {}

    # Determine per-run loop counts so we can normalise
    run_loop_counts: dict[str, int] = {}
    for path in all_loop_paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue
        run_id = relative.parts[0]
        idx = _parse_loop_ordinal(path.stem)
        if idx is not None:
            run_loop_counts[run_id] = min(
                max_loop_count,
                max(run_loop_counts.get(run_id, 0), idx + 1),
            )

    condition_names: list[str] = []
    unlabeled: list[tuple[Path, str, int, int]] = []  # path, condition, loop_idx, loop_count

    for path in all_loop_paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(
                "Loop images must be stored under data/images/<run-id>/.../<NNth Loop>.png"
            )
        run_id = relative.parts[0]
        if run_id not in json_index:
            raise FileNotFoundError(f"No JSON file found for image run: {run_id}")
        if run_id not in payload_cache:
            payload_cache[run_id] = _read_json_payload(json_index[run_id])

        loop_idx = _parse_loop_ordinal(path.stem)
        if loop_idx is None:
            continue  # should not happen given the filter above
        if loop_idx >= max_loop_count:
            continue  # index out of range for the embedding table / sequence conditioner
        loop_count = run_loop_counts.get(run_id, 1)

        condition = _label_from_payload(payload_cache[run_id], label_source)
        condition_names.append(condition)
        unlabeled.append((path, condition, loop_idx, loop_count))

    if not condition_names:
        raise ValueError(f"No valid loop images with JSON metadata found under: {root}")

    condition_to_idx = {name: idx for idx, name in enumerate(sorted(set(condition_names)))}
    samples: list[GeneratedImageSample] = []
    for path, condition, loop_idx, loop_count in unlabeled:
        loop_start, loop_end = _loop_normalised_bounds(loop_idx, loop_count)
        samples.append(
            GeneratedImageSample(
                path=path,
                condition=condition,
                label=condition_to_idx[condition],
                loop_idx=loop_idx,
                loop_count=loop_count,
                loop_start=loop_start,
                loop_end=loop_end,
            )
        )
    return samples, condition_to_idx


def _normalize_image_array(image: Image.Image, channels: int) -> np.ndarray:
    if channels == 1:
        array = np.asarray(image.convert("L"), dtype=np.float32)[None, ...]
    elif channels == 3:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    elif channels == 4:
        array = np.asarray(image.convert("RGBA"), dtype=np.float32).transpose(2, 0, 1)
    else:
        raise ValueError("channels must be one of 1, 3, or 4")
    return array / 127.5 - 1.0


def _fit_image(
    image: Image.Image,
    image_size: int,
    fit_mode: FitMode,
    channels: int,
) -> Image.Image:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    background_mode = "L" if channels == 1 else "RGBA" if channels == 4 else "RGB"
    if fit_mode == "resize":
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR).convert(background_mode)
    if fit_mode == "reshape":
        converted = image.convert(background_mode)
        reshaped_channels = 1 if channels == 1 else channels
        flat = np.asarray(converted, dtype=np.uint8).reshape(-1, reshaped_channels)
        pixel_count = int(flat.shape[0])
        side = math.isqrt(pixel_count)
        if side * side != pixel_count:
            raise ValueError(
                "reshape fit mode requires width*height to be a perfect square "
                f"(got {converted.width}x{converted.height}={pixel_count})"
            )
        reshaped = flat.reshape(side, side, reshaped_channels)
        if channels == 1:
            return Image.fromarray(reshaped[:, :, 0])
        return Image.fromarray(reshaped)
    if fit_mode == "height-flatten":
        converted = image.convert("RGB")
        img_width, img_height = ImgConfig().img_size
        if img_width != img_height:
            raise ValueError(
                "height-flatten fit mode requires square ImgConfig.img_size for square output; "
                f"got {ImgConfig().img_size}"
            )
        if converted.width % img_width != 0 or converted.height % img_height != 0:
            raise ValueError(
                "height-flatten fit mode requires dimensions to be multiples of "
                f"ImgConfig.img_size={ImgConfig().img_size}; got {converted.width}x{converted.height}"
            )
        source = np.asarray(converted, dtype=np.uint8)
        rows = converted.height // img_height
        cols = converted.width // img_width
        block_count = rows * cols
        square_blocks = math.isqrt(block_count)
        if square_blocks * square_blocks != block_count:
            raise ValueError(
                "height-flatten fit mode requires the number of ImgConfig-sized blocks "
                f"to be a perfect square (got {block_count})"
            )
        # Flatten by ImgConfig-sized blocks, then reshape block order into a square grid.
        blocks = source.reshape(rows, img_height, cols, img_width, 3).transpose(0, 2, 1, 3, 4)
        flattened_blocks = blocks.reshape(block_count, img_height, img_width, 3)
        squared_blocks = flattened_blocks.reshape(
            square_blocks, square_blocks, img_height, img_width, 3
        )
        reshaped = squared_blocks.transpose(0, 2, 1, 3, 4).reshape(
            square_blocks * img_height,
            square_blocks * img_width,
            3,
        )
        return Image.fromarray(reshaped)
    if fit_mode != "pad":
        raise ValueError(f"Unsupported fit mode: {fit_mode}")

    if channels == 1:
        fill = 255
    elif channels == 4:
        fill = (255, 255, 255, 255)
    else:
        fill = (255, 255, 255)
    fitted = image.convert(background_mode)
    side = max(fitted.width, fitted.height)
    canvas = Image.new(background_mode, (side, side), fill)
    offset = ((side - fitted.width) // 2, (side - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _ensure_square_batch(images: Tensor) -> Tensor:
    if images.ndim != 4:
        raise ValueError(f"images must be a 4D tensor (N,C,H,W), got shape {tuple(images.shape)}")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    side = max(height, width)
    if height == width:
        return images
    return F.interpolate(
        images,
        size=(side, side),
        mode="bilinear",
        align_corners=False,
    )


def cleanup_torch_resources(
    objects: dict[str, object] | None = None,
    *,
    synchronize_mps: bool = True,
) -> None:
    """Release common training-time references and clear backend caches."""

    if objects is not None:
        objects.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.mps.is_available():
        if synchronize_mps:
            torch.mps.synchronize()
        torch.mps.empty_cache()


class GeneratedImageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """
    Dataset for images generated by ``RGBImgMaker``.

    Each item returns ``(image_tensor, label_tensor, loop_meta_tensor)`` where:

    * ``image_tensor``    – float32 (C, H, W) normalised to ``[-1, 1]``.
    * ``label_tensor``    – long scalar condition id.
    * ``loop_meta_tensor``– float32 (4,) = [loop_idx, loop_count, loop_start,
                            loop_end].  Defaults to ``[0, 1, 0, 1]`` for
                            ``message.png`` images that carry no loop context.
    """

    def __init__(
        self,
        root: Path | str,
        json_root: Path | str = Path("output/json"),
        image_size: int = 64,
        channels: int = 3,
        fit_mode: FitMode = "pad",
        condition_mode: ConditionMode = "json-step",
        label_source: LabelSource = "final-hash",
        max_images: int | None = None,
        use_loop_images: bool = False,
        max_loop_count: int = 64,
    ) -> None:
        self.root = Path(root)
        self.json_root = Path(json_root)
        self.image_size = image_size
        self.fit_mode = fit_mode
        self.source_channels = channels
        self.channels = channels
        self.condition_mode = condition_mode
        self.label_source = label_source
        if use_loop_images:
            self.samples, self.condition_to_idx = discover_loop_image_samples(
                self.root,
                json_root=json_root,
                label_source=label_source,
                max_images=max_images,
                max_loop_count=max_loop_count,
            )
        else:
            self.samples, self.condition_to_idx = discover_generated_image_samples(
                self.root,
                condition_mode=condition_mode,
                json_root=json_root,
                label_source=label_source,
                max_images=max_images,
            )
        self.idx_to_condition = {idx: name for name, idx in self.condition_to_idx.items()}

    @property
    def num_conditions(self) -> int:
        return len(self.condition_to_idx)

    @property
    def condition_names(self) -> list[str]:
        return [self.idx_to_condition[idx] for idx in range(self.num_conditions)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            fitted = _fit_image(image, self.image_size, self.fit_mode, self.source_channels)
            array = _normalize_image_array(fitted, self.channels)
        image_tensor = torch.from_numpy(array).to(dtype=torch.float32)
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        loop_meta = torch.tensor(
            [float(sample.loop_idx), float(sample.loop_count), sample.loop_start, sample.loop_end],
            dtype=torch.float32,
        )
        return image_tensor, label_tensor, loop_meta


class IndexedGeneratedImageDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Wrapper dataset that exposes sample index for batch metadata export."""

    def __init__(self, base: GeneratedImageDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image, label, loop_meta = self.base[index]
        return image, label, loop_meta, torch.tensor(index, dtype=torch.long)


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding used by the denoiser."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.dim // 2
        if half_dim == 0:
            return torch.zeros((timesteps.shape[0], 0), device=timesteps.device)
        scale = math.log(10_000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        )
        args = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ConditionalResBlock(nn.Module):
    """Residual block modulated by time and condition embeddings."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal conditioning modules
# ──────────────────────────────────────────────────────────────────────────────

class LoopSinusoidalConditioner(nn.Module):
    """
    Method 1 – Sinusoidal positional encoding for loop index.

    Replaces the discrete ``nn.Embedding`` with the same sinusoidal scheme used
    for diffusion timesteps so that adjacent loops share similar embeddings and
    the model can smoothly interpolate between positions.

    ``loop_meta[:, 0]`` is the loop index (converted to ``long``).
    """

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, loop_meta: Tensor) -> Tensor:
        loop_idx = loop_meta[:, 0].long()
        return self.emb(loop_idx)


class LoopStructuredConditioner(nn.Module):
    """
    Method 2 – Structured 3-part conditioning vector.

    Combines three projections, each mapped to ``time_dim`` and summed:

    * **idx_emb**    – sinusoidal PE of ``loop_idx``   (sequence position).
    * **start_proj** – MLP of ``loop_start`` ∈ [0, 1] (normalised start).
    * **end_proj**   – MLP of ``loop_end``   ∈ [0, 1] (normalised end).

    ``loop_meta`` layout: ``[:, 0]``=loop_idx  ``[:, 1]``=loop_count
    ``[:, 2]``=loop_start  ``[:, 3]``=loop_end
    """

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.idx_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.start_proj = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.end_proj = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, loop_meta: Tensor) -> Tensor:
        loop_idx   = loop_meta[:, 0].long()
        loop_start = loop_meta[:, 2:3]   # (B, 1)
        loop_end   = loop_meta[:, 3:4]   # (B, 1)
        return self.idx_emb(loop_idx) + self.start_proj(loop_start) + self.end_proj(loop_end)


def _valid_num_heads(dim: int, preferred: int = 4) -> int:
    """Return the largest even divisor of ``dim`` that is ≤ ``preferred``."""
    for h in range(preferred, 0, -1):
        if dim % h == 0:
            return h
    return 1


class LoopSequenceConditioner(nn.Module):
    """
    Method 3 – Transformer-sequence conditioner.

    Maintains one learnable token per loop position (up to ``max_loop_count``).
    A small Transformer encoder attends over the *entire* sequence so that each
    position is contextualised by all other positions (bidirectional).

    On each forward pass the full sequence is re-encoded (cheap for ≤ 64
    tokens) so gradients flow through the transformer weights.  The output at
    ``loop_idx`` is the conditioning vector for that sample.

    ``loop_meta`` layout: ``[:, 0]``=loop_idx  ``[:, 1]``=loop_count
    ``[:, 2]``=loop_start  ``[:, 3]``=loop_end
    """

    def __init__(
        self,
        max_loop_count: int,
        time_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.max_loop_count = max_loop_count
        self.loop_tokens = nn.Embedding(max_loop_count, time_dim)
        heads = _valid_num_heads(time_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=time_dim,
            nhead=heads,
            dim_feedforward=time_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

    def forward(self, loop_meta: Tensor) -> Tensor:
        loop_idx = loop_meta[:, 0].long()
        # Re-encode the full sequence on every forward pass (gradient-safe).
        seq = self.loop_tokens.weight.unsqueeze(0)      # (1, max_loop_count, time_dim)
        ctx = self.transformer(seq).squeeze(0)          # (max_loop_count, time_dim)
        return self.proj(ctx[loop_idx])                 # (B, time_dim)


class ConditionalUNet(nn.Module):
    """Small conditional U-Net that predicts DDPM noise."""

    def __init__(
        self,
        in_channels: int,
        num_conditions: int,
        base_channels: int = 64,
        time_dim: int = 256,
        temporal_conditioning: TemporalConditioningMode = "class",
        max_loop_count: int = 64,
    ) -> None:
        super().__init__()
        if num_conditions <= 0:
            raise ValueError("num_conditions must be positive")

        self.in_channels = in_channels
        self.num_conditions = num_conditions
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.temporal_conditioning = temporal_conditioning
        self.max_loop_count = max_loop_count

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Build the condition embedding based on the requested temporal mode.
        if temporal_conditioning == "class":
            self.condition_embedding: nn.Module = nn.Embedding(num_conditions, time_dim)
        elif temporal_conditioning == "loop-sinusoidal":
            self.condition_embedding = LoopSinusoidalConditioner(time_dim)
        elif temporal_conditioning == "loop-structured":
            self.condition_embedding = LoopStructuredConditioner(time_dim)
        elif temporal_conditioning == "loop-sequence":
            self.condition_embedding = LoopSequenceConditioner(
                max_loop_count=max_loop_count,
                time_dim=time_dim,
            )
        else:
            raise ValueError(f"Unsupported temporal_conditioning: {temporal_conditioning!r}")

        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ConditionalResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down2 = ConditionalResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.mid = ConditionalResBlock(base_channels * 4, base_channels * 4, time_dim)

        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up1 = ConditionalResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up2 = ConditionalResBlock(base_channels * 2, base_channels, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def _embedding(
        self,
        timesteps: Tensor,
        labels: Tensor,
        loop_meta: Tensor | None = None,
    ) -> Tensor:
        time_emb = self.time_embedding(timesteps)
        if self.temporal_conditioning == "class":
            cond_emb = self.condition_embedding(labels)
        else:
            if loop_meta is None:
                raise ValueError(
                    f"loop_meta is required for temporal_conditioning={self.temporal_conditioning!r}"
                )
            cond_emb = self.condition_embedding(loop_meta)
        return time_emb + cond_emb

    @staticmethod
    def _match_spatial(x: Tensor, target: Tensor) -> Tensor:
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        loop_meta: Tensor | None = None,
    ) -> Tensor:
        emb = self._embedding(timesteps, labels, loop_meta)
        x0 = self.input(x)
        x1 = self.down1(x0, emb)
        x2 = self.downsample1(x1)
        x3 = self.down2(x2, emb)
        x4 = self.downsample2(x3)
        xm = self.mid(x4, emb)

        u1 = self._match_spatial(self.upsample1(xm), x3)
        u1 = self.up1(torch.cat([u1, x3], dim=1), emb)
        u2 = self._match_spatial(self.upsample2(u1), x1)
        u2 = self.up2(torch.cat([u2, x1], dim=1), emb)
        return self.output(u2)


class DDPMNoiseScheduler:
    """Forward and reverse diffusion coefficients for DDPM."""

    def __init__(
        self,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
        betas: Tensor | np.ndarray | list[float] | None = None,
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device(device)
        if betas is None:
            if timesteps <= 0:
                raise ValueError("timesteps must be positive")
            self.timesteps = timesteps
            self.betas = torch.linspace(
                beta_start,
                beta_end,
                timesteps,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            beta_tensor = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
            if beta_tensor.ndim != 1 or beta_tensor.numel() == 0:
                raise ValueError("betas must be a non-empty one-dimensional array")
            if torch.any(beta_tensor <= 0.0) or torch.any(beta_tensor >= 1.0):
                raise ValueError("all beta values must be in the open interval (0, 1)")
            self.timesteps = int(beta_tensor.numel())
            self.betas = beta_tensor
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32, device=self.device), self.alpha_bars[:-1]],
            dim=0,
        )
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

    def to(self, device: torch.device | str) -> DDPMNoiseScheduler:
        return DDPMNoiseScheduler(
            timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=device,
            betas=self.betas.detach().cpu(),
        )

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, target_ndim: int) -> Tensor:
        out = values.gather(0, timesteps)
        return out.reshape(timesteps.shape[0], *([1] * (target_ndim - 1)))

    def q_sample(self, x0: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bars = self._extract(self.sqrt_alpha_bars, timesteps, x0.ndim)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, x0.ndim)
        return sqrt_alpha_bars * x0 + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: Tensor,
        step: int,
        labels: Tensor,
        loop_meta: Tensor | None = None,
    ) -> Tensor:
        timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        beta_t = self._extract(self.betas, timesteps, x.ndim)
        alpha_t = self._extract(self.alphas, timesteps, x.ndim)
        alpha_bar_t = self._extract(self.alpha_bars, timesteps, x.ndim)
        pred_noise = model(x, timesteps, labels, loop_meta)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )
        if step == 0:
            return mean
        variance = self._extract(self.posterior_variance, timesteps, x.ndim)
        return mean + torch.sqrt(variance) * torch.randn_like(x)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        labels: Tensor,
        loop_meta: Tensor | None = None,
    ) -> Tensor:
        x = torch.randn(shape, device=labels.device)
        model.eval()
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step, labels, loop_meta)
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_with_trace(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        labels: Tensor,
        trace_timesteps: Iterable[int],
        loop_meta: Tensor | None = None,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        x = torch.randn(shape, device=labels.device)
        model.eval()
        trace_set = {int(step) for step in trace_timesteps}
        traces: dict[int, Tensor] = {self.timesteps: x.clamp(-1.0, 1.0).detach().cpu()}
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step, labels, loop_meta)
            if step in trace_set:
                traces[step] = x.clamp(-1.0, 1.0).detach().cpu()
        return x.clamp(-1.0, 1.0), traces


def _as_1d_beta_array(values: object, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError(f"{name} values must be in the open interval (0, 1)")
    return arr


def _load_beta_values(path: Path | str) -> np.ndarray:
    beta_path = Path(path)
    if not beta_path.exists():
        raise FileNotFoundError(f"Beta values file does not exist: {beta_path}")

    suffix = beta_path.suffix.lower()
    if suffix == ".npy":
        return _as_1d_beta_array(np.load(beta_path), name="betas")
    if suffix == ".npz":
        loaded = np.load(beta_path)
        key = "betas" if "betas" in loaded else loaded.files[0]
        return _as_1d_beta_array(loaded[key], name=f"betas[{key}]")

    text = beta_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Beta values file is empty: {beta_path}")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in beta values file: {beta_path} ({exc.msg} at line {exc.lineno}, column {exc.colno})"
            ) from exc
        values = payload["betas"] if isinstance(payload, dict) and "betas" in payload else payload
        return _as_1d_beta_array(values, name="betas")

    values = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float64)
    return _as_1d_beta_array(values, name="betas")


def _hash_approach_beta_candidates(config: ConditionalDiffusionTrainConfig) -> tuple[np.ndarray, np.ndarray]:
    analyzer = Analyze(config.json_root, step_name=config.beta_schedule_step)
    summary = analyzer.summarize_beta_schedules(step_name=config.beta_schedule_step)
    beta_scheduler = BetaScheduler(
        beta_min=config.beta_start,
        beta_max=config.beta_end,
    )
    approach1 = _as_1d_beta_array(
        beta_scheduler.approach1(summary.mean).rescaled_candidate,
        name="hash-approach1 betas",
    )
    approach2 = _as_1d_beta_array(
        beta_scheduler.approach2(summary.mean).candidate,
        name="hash-approach2 betas",
    )
    return approach1, approach2


def build_beta_schedule(config: ConditionalDiffusionTrainConfig) -> np.ndarray | None:
    """
    Build optional custom betas for DDPMNoiseScheduler.

    Returns None for standard linear schedule unless timesteps='auto'.
    """

    if config.beta_schedule == "linear":
        if config.timesteps == "auto":
            approach1_betas, approach2_betas = _hash_approach_beta_candidates(config)
            if approach1_betas.size != approach2_betas.size:
                raise ValueError(
                    "Hash approach schedule length mismatch: "
                    f"approach1={approach1_betas.size}, approach2={approach2_betas.size}"
                )
            return _as_1d_beta_array(
                np.linspace(
                    config.beta_start,
                    config.beta_end,
                    int(approach1_betas.size),
                    dtype=np.float64,
                ),
                name="betas",
            )
        return None
    if config.beta_schedule == "file":
        if config.beta_values_path is None:
            raise ValueError("beta_values_path is required when beta_schedule='file'")
        return _load_beta_values(config.beta_values_path)

    approach1_betas, approach2_betas = _hash_approach_beta_candidates(config)
    if config.beta_schedule == "hash-approach1":
        betas = approach1_betas
    elif config.beta_schedule == "hash-approach2":
        betas = approach2_betas
    else:
        raise ValueError(f"Unsupported beta schedule: {config.beta_schedule}")

    return _as_1d_beta_array(betas, name="betas")


def _parse_timesteps_arg(value: str) -> int | Literal["auto"]:
    text = value.strip().lower()
    if text == "auto":
        return "auto"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timesteps must be a positive integer or 'auto'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timesteps must be a positive integer or 'auto'")
    return parsed


def save_beta_schedule(
    scheduler: DDPMNoiseScheduler,
    config: ConditionalDiffusionTrainConfig,
) -> Path:
    path = config.output_dir / "beta_schedule.json"
    payload = {
        "mode": config.beta_schedule,
        "timesteps": scheduler.timesteps,
        "beta_start": float(scheduler.betas[0].detach().cpu()),
        "beta_end": float(scheduler.betas[-1].detach().cpu()),
        "betas": [float(value) for value in scheduler.betas.detach().cpu().tolist()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_preprocess_summary(dataset: GeneratedImageDataset, fit_mode: FitMode) -> None:
    sample_path = dataset.samples[0].path
    with Image.open(sample_path) as sample_image:
        src_w, src_h = sample_image.width, sample_image.height
    out_c, out_h, out_w = (int(v) for v in dataset[0][0].shape)
    if fit_mode == "height-flatten":
        img_w, img_h = ImgConfig().img_size
        block_count = (src_w // img_w) * (src_h // img_h)
        square_blocks = math.isqrt(block_count)
        print(
            f"[reshape] mode=height-flatten source={src_w}x{src_h} "
            f"img_size={img_w}x{img_h} blocks={block_count} ({square_blocks}x{square_blocks}) "
            f"output={out_w}x{out_h} channels={out_c}"
        )
        return
    print(
        f"[reshape] mode={fit_mode} source={src_w}x{src_h} "
        f"output={out_w}x{out_h} channels={out_c}"
    )


def _denormalize_images(images: Tensor) -> Tensor:
    return ((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)


def _image_from_tensor(image_tensor: Tensor) -> Image.Image:
    channels = int(image_tensor.shape[0])
    array = image_tensor.numpy()
    if channels == 1:
        return Image.fromarray(array[0])
    return Image.fromarray(array.transpose(1, 2, 0))


def save_image_grid(
    images: Tensor,
    labels: Tensor,
    condition_names: list[str],
    path: Path,
    *,
    json_path: Path | None = None,
) -> None:
    """Save image outputs as individual PNG files (one per sample).

    Args:
        path: Destination PNG path.
        json_path: If given, save the labels sidecar JSON here instead of
            alongside *path*.  Useful when PNG and JSON should live in
            separate directories (e.g. process-trace output).
    """

    images = _denormalize_images(images)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    if images.shape[0] == 1:
        _image_from_tensor(images[0]).save(path)
        saved_files.append(path.name)
    else:
        stem = path.stem
        suffix = path.suffix
        for idx, image_tensor in enumerate(images):
            file_path = path.with_name(f"{stem}_{idx:03d}{suffix}")
            _image_from_tensor(image_tensor).save(file_path)
            saved_files.append(file_path.name)

    labels_path = json_path if json_path is not None else path.with_suffix(".labels.json")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    label_values = labels.detach().cpu().tolist()
    labels_path.write_text(
        json.dumps(
            [
                {
                    "index": idx,
                    "file": saved_files[idx],
                    "label": int(label),
                    "condition": condition_names[int(label)],
                }
                for idx, label in enumerate(label_values)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def _reference_images_for_labels(dataset: GeneratedImageDataset, labels: Tensor) -> Tensor:
    first_index_by_label: dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        first_index_by_label.setdefault(int(sample.label), index)

    reference_images: list[Tensor] = []
    for label in labels.detach().cpu().tolist():
        label_int = int(label)
        if label_int not in first_index_by_label:
            raise ValueError(f"Label {label_int} not found in dataset")
        image = dataset[first_index_by_label[label_int]][0]  # (image, label, loop_meta)
        reference_images.append(image)
    return torch.stack(reference_images)


def save_source_generated_grid(
    source_images: Tensor,
    generated_images: Tensor,
    labels: Tensor,
    condition_names: list[str],
    path: Path,
) -> None:
    if source_images.shape != generated_images.shape:
        raise ValueError(
            "source_images and generated_images must have the same shape, "
            f"got {tuple(source_images.shape)} and {tuple(generated_images.shape)}"
        )
    source = _denormalize_images(source_images)
    generated = _denormalize_images(generated_images)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_files: list[dict[str, str]] = []
    if source.shape[0] == 1:
        src_path = path.with_name(f"{path.stem}.source{path.suffix}")
        gen_path = path.with_name(f"{path.stem}.generated{path.suffix}")
        _image_from_tensor(source[0]).save(src_path)
        _image_from_tensor(generated[0]).save(gen_path)
        saved_files.append({"source": src_path.name, "generated": gen_path.name})
    else:
        stem = path.stem
        suffix = path.suffix
        for idx in range(source.shape[0]):
            src_path = path.with_name(f"{stem}_{idx:03d}.source{suffix}")
            gen_path = path.with_name(f"{stem}_{idx:03d}.generated{suffix}")
            _image_from_tensor(source[idx]).save(src_path)
            _image_from_tensor(generated[idx]).save(gen_path)
            saved_files.append({"source": src_path.name, "generated": gen_path.name})

    labels_path = path.with_suffix(".labels.json")
    label_values = labels.detach().cpu().tolist()
    labels_path.write_text(
        json.dumps(
            [
                {
                    "index": idx,
                    "label": int(label),
                    "condition": condition_names[int(label)],
                    "files": saved_files[idx],
                }
                for idx, label in enumerate(label_values)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def save_train_batch_grid(
    images: Tensor,
    labels: Tensor,
    condition_names: list[str],
    output_dir: Path,
    step: int,
    sample_records: list[dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"step_{step:06d}.png"
    save_image_grid(images, labels, condition_names, path)
    metadata_path = output_dir / f"step_{step:06d}.batch.json"
    metadata_path.write_text(
        json.dumps({"step": step, "samples": sample_records}, indent=2),
        encoding="utf-8",
    )
    return path


def trace_timesteps(timesteps: int, trace_steps: int) -> list[int]:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if trace_steps <= 0:
        return []
    count = min(timesteps, trace_steps)
    values = np.linspace(0, timesteps - 1, num=count, dtype=np.int64)
    return sorted({int(value) for value in values.tolist()})


def _save_forward_process_trace(
    scheduler: DDPMNoiseScheduler,
    images: Tensor,
    labels: Tensor,
    condition_names: list[str],
    output_dir: Path,
) -> list[Path]:
    png_dir = output_dir / "png"
    json_dir = output_dir / "json"
    png_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    print(f"[forward-trace] saving x0 + {scheduler.timesteps} noising steps to {output_dir}")

    x0_path = png_dir / "x0.png"
    save_image_grid(
        images.detach().cpu(), labels.detach().cpu(), condition_names, x0_path,
        json_path=json_dir / "x0.labels.json",
    )
    saved_paths.append(x0_path)
    print(f"[forward-trace] saved x0: {x0_path}")

    noise = torch.randn_like(images)
    for step in range(scheduler.timesteps):
        timesteps = torch.full((images.shape[0],), step, device=images.device, dtype=torch.long)
        noised = scheduler.q_sample(images, timesteps, noise)
        path = png_dir / f"t_{step:06d}.png"
        save_image_grid(
            noised, labels, condition_names, path,
            json_path=json_dir / f"t_{step:06d}.labels.json",
        )
        saved_paths.append(path)
        print(f"[forward-trace] step={step:06d} path={path}")

    print(f"[forward-trace] completed: {len(saved_paths)} files")
    return saved_paths


def _save_reverse_process_trace(
    scheduler: DDPMNoiseScheduler,
    model: nn.Module,
    labels: Tensor,
    condition_names: list[str],
    output_dir: Path,
    sample_shape: tuple[int, int, int, int],
    loop_meta: Tensor | None = None,
) -> list[Path]:
    png_dir = output_dir / "png"
    json_dir = output_dir / "json"
    png_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    print(f"[reverse-trace] saving xT + {scheduler.timesteps} denoising steps to {output_dir}")
    _, traces = scheduler.sample_with_trace(
        model,
        sample_shape,
        labels,
        range(scheduler.timesteps),
        loop_meta=loop_meta,
    )

    for step in sorted(traces.keys(), reverse=True):
        filename = "xT_noise.png" if step == scheduler.timesteps else f"t_{step:06d}.png"
        path = png_dir / filename
        json_name = filename.replace(".png", ".labels.json")
        save_image_grid(
            traces[step], labels.detach().cpu(), condition_names, path,
            json_path=json_dir / json_name,
        )
        saved_paths.append(path)
        if step == scheduler.timesteps:
            print(f"[reverse-trace] saved xT: {path}")
        else:
            print(f"[reverse-trace] step={step:06d} path={path}")
    print(f"[reverse-trace] completed: {len(saved_paths)} files")
    return saved_paths


def save_process_traces(
    model: nn.Module,
    scheduler: DDPMNoiseScheduler,
    dataset: GeneratedImageDataset,
    config: ConditionalDiffusionTrainConfig,
    device: torch.device,
) -> dict[str, list[Path]]:
    if config.trace_sample_count <= 0:
        raise ValueError("trace_sample_count must be positive")

    sample_count = min(config.trace_sample_count, len(dataset))
    images = []
    labels = []
    loop_metas = []
    for index in range(sample_count):
        image, label, loop_meta = dataset[index]
        images.append(image)
        labels.append(int(label))
        loop_metas.append(loop_meta)

    image_batch = torch.stack(images).to(device)
    label_batch = torch.tensor(labels, dtype=torch.long, device=device)
    loop_meta_batch = torch.stack(loop_metas).to(device)
    trace_dir = config.output_dir / "process_traces"
    sample_shape = (image_batch.shape[0], dataset.channels, image_batch.shape[-2], image_batch.shape[-1])

    forward_paths = _save_forward_process_trace(
        scheduler,
        image_batch,
        label_batch,
        dataset.condition_names,
        trace_dir / "forward",
    )
    reverse_paths = _save_reverse_process_trace(
        scheduler,
        model,
        label_batch,
        dataset.condition_names,
        trace_dir / "reverse",
        sample_shape,
        loop_meta=loop_meta_batch,
    )
    return {"forward": forward_paths, "reverse": reverse_paths}


def _condition_tensors_for_sampling(
    dataset: GeneratedImageDataset,
    sample_count: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return ``(labels, loop_meta)`` tensors for inference sampling.

    Labels cycle over all condition ids.  ``loop_meta`` is built from the
    first dataset sample that matches each label so the temporal context
    matches real data seen during training.
    """
    condition_ids = list(range(dataset.num_conditions))
    label_list = [label for _, label in zip(range(sample_count), cycle(condition_ids))]

    # Build a fast lookup: label → first matching sample
    first_by_label: dict[int, GeneratedImageSample] = {}
    for sample in dataset.samples:
        first_by_label.setdefault(sample.label, sample)

    loop_meta_list: list[list[float]] = []
    for lbl in label_list:
        s = first_by_label.get(lbl)
        if s is not None:
            loop_meta_list.append(
                [float(s.loop_idx), float(s.loop_count), s.loop_start, s.loop_end]
            )
        else:
            loop_meta_list.append([0.0, 1.0, 0.0, 1.0])

    labels_tensor = torch.tensor(label_list, dtype=torch.long, device=device)
    loop_meta_tensor = torch.tensor(loop_meta_list, dtype=torch.float32, device=device)
    return labels_tensor, loop_meta_tensor


def _checkpoint_payload(
    model: ConditionalUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    dataset: GeneratedImageDataset,
    config: ConditionalDiffusionTrainConfig,
) -> dict[str, object]:
    config_dict = asdict(config)
    config_dict["data_root"] = str(config.data_root)
    config_dict["json_root"] = str(config.json_root)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["beta_values_path"] = (
        None if config.beta_values_path is None else str(config.beta_values_path)
    )
    return {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "condition_to_idx": dataset.condition_to_idx,
        "config": config_dict,
        "model_args": {
            "in_channels": model.in_channels,
            "num_conditions": model.num_conditions,
            "base_channels": model.base_channels,
            "time_dim": model.time_dim,
            "temporal_conditioning": model.temporal_conditioning,
            "max_loop_count": model.max_loop_count,
        },
    }


def save_checkpoint(
    model: ConditionalUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    dataset: GeneratedImageDataset,
    config: ConditionalDiffusionTrainConfig,
) -> Path:
    path = config.output_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, step, loss, dataset, config), path)
    return path


def resolve_train_steps(
    dataset_size: int,
    batch_size: int,
    train_steps: int,
    epochs: int | None = None,
) -> int:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if epochs is None:
        return train_steps
    if epochs <= 0:
        raise ValueError("epochs must be positive when provided")
    return math.ceil(dataset_size / batch_size) * epochs


def train_conditional_diffusion(
    config: ConditionalDiffusionTrainConfig,
) -> dict[str, Path | float | int | None]:
    """Train a conditional DDPM on generated images."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = GeneratedImageDataset(
        root=config.data_root,
        json_root=config.json_root,
        image_size=config.image_size,
        channels=config.channels,
        fit_mode=config.fit_mode,
        condition_mode=config.condition_mode,
        label_source=config.label_source,
        max_images=config.max_images,
        use_loop_images=config.use_loop_images,
        max_loop_count=config.max_loop_count,
    )
    _print_preprocess_summary(dataset, config.fit_mode)
    indexed_dataset = IndexedGeneratedImageDataset(dataset)
    dataloader = DataLoader(
        indexed_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    loader_iter: Iterable[tuple[Tensor, Tensor, Tensor, Tensor]] = cycle(dataloader)
    effective_train_steps = resolve_train_steps(
        dataset_size=len(dataset),
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        epochs=config.epochs,
    )

    model = ConditionalUNet(
        in_channels=dataset.channels,
        num_conditions=dataset.num_conditions,
        base_channels=config.base_channels,
        time_dim=config.time_dim,
        temporal_conditioning=config.temporal_conditioning,
        max_loop_count=config.max_loop_count,
    ).to(device)
    sample_image_shape = tuple(int(value) for value in dataset[0][0].shape)
    sample_channels, sample_height, sample_width = sample_image_shape
    custom_betas = build_beta_schedule(config)
    scheduler = DDPMNoiseScheduler(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
        betas=custom_betas,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "condition_to_idx.json").write_text(
        json.dumps(dataset.condition_to_idx, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config_dict = asdict(config)
    config_dict["data_root"] = str(config.data_root)
    config_dict["json_root"] = str(config.json_root)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["beta_values_path"] = (
        None if config.beta_values_path is None else str(config.beta_values_path)
    )
    (config.output_dir / "train_config.json").write_text(
        json.dumps(config_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    beta_schedule_path = save_beta_schedule(scheduler, config)

    last_loss = math.nan
    print(
        f"dataset={len(dataset)} images conditions={dataset.num_conditions} "
        f"steps={effective_train_steps} epochs={config.epochs} "
        f"device={device} beta_schedule={config.beta_schedule} "
        f"diffusion_timesteps={scheduler.timesteps} "
        f"temporal_conditioning={config.temporal_conditioning} "
        f"sample_image_shape={sample_image_shape} output={config.output_dir}"
    )

    train_batch_dir: Path | None = None
    for step in range(1, effective_train_steps + 1):
        images, labels, loop_meta, indices = next(loader_iter)
        images = images.to(device=device, non_blocking=True)
        if config.fit_mode != "height-flatten":
            images = _ensure_square_batch(images)
        labels = labels.to(device=device, non_blocking=True)
        loop_meta = loop_meta.to(device=device, non_blocking=True)
        if config.save_train_batches_every > 0 and step % config.save_train_batches_every == 0:
            train_batch_dir = config.output_dir / "train_batches"
            sample_records = []
            for idx_tensor, label_tensor in zip(indices.detach().cpu(), labels.detach().cpu()):
                sample_index = int(idx_tensor)
                sample = dataset.samples[sample_index]
                sample_records.append(
                    {
                        "dataset_index": sample_index,
                        "path": str(sample.path),
                        "condition": sample.condition,
                        "label": int(label_tensor),
                    }
                )
            train_batch_path = save_train_batch_grid(
                images.detach().cpu(),
                labels.detach().cpu(),
                dataset.condition_names,
                train_batch_dir,
                step,
                sample_records=sample_records,
            )
            print(f"saved train batch: {train_batch_path}")
            print(f"saved train batch metadata: {train_batch_path.with_suffix('.batch.json')}")
        timesteps = torch.randint(0, scheduler.timesteps, (images.shape[0],), device=device)
        noise = torch.randn_like(images)
        noised = scheduler.q_sample(images, timesteps, noise)
        pred_noise = model(noised, timesteps, labels, loop_meta)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step == 1 or step % config.log_every == 0 or step == effective_train_steps:
            print(f"step={step:06d} loss={last_loss:.6f}")

        if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
            checkpoint_path = save_checkpoint(model, optimizer, step, last_loss, dataset, config)
            print(f"saved checkpoint: {checkpoint_path}")

        if config.sample_every > 0 and step % config.sample_every == 0:
            labels_for_sample, loop_meta_for_sample = _condition_tensors_for_sampling(
                dataset, config.sample_count, device
            )
            samples = scheduler.sample(
                model,
                (
                    labels_for_sample.shape[0],
                    sample_channels,
                    sample_height,
                    sample_width,
                ),
                labels_for_sample,
                loop_meta=loop_meta_for_sample,
            )
            sample_path = config.output_dir / "samples" / f"step_{step:06d}.png"
            save_image_grid(samples, labels_for_sample, dataset.condition_names, sample_path)
            reference_images = _reference_images_for_labels(dataset, labels_for_sample)
            paired_sample_path = config.output_dir / "samples" / f"step_{step:06d}.with_source.png"
            save_source_generated_grid(
                reference_images,
                samples,
                labels_for_sample,
                dataset.condition_names,
                paired_sample_path,
            )
            print(f"saved samples: {sample_path}")
            print(f"saved source+generated samples: {paired_sample_path}")
            model.train()

    final_checkpoint = save_checkpoint(
        model,
        optimizer,
        effective_train_steps,
        last_loss,
        dataset,
        config,
    )
    labels_for_sample, loop_meta_for_sample = _condition_tensors_for_sampling(
        dataset, config.sample_count, device
    )
    samples = scheduler.sample(
        model,
        (
            labels_for_sample.shape[0],
            sample_channels,
            sample_height,
            sample_width,
        ),
        labels_for_sample,
        loop_meta=loop_meta_for_sample,
    )
    final_sample_path = config.output_dir / "samples" / "final.png"
    save_image_grid(samples, labels_for_sample, dataset.condition_names, final_sample_path)
    final_source_path = config.output_dir / "samples" / "final.source.png"
    reference_images = _reference_images_for_labels(dataset, labels_for_sample)
    save_image_grid(reference_images, labels_for_sample, dataset.condition_names, final_source_path)
    final_with_source_path = config.output_dir / "samples" / "final.with_source.png"
    save_source_generated_grid(
        reference_images,
        samples,
        labels_for_sample,
        dataset.condition_names,
        final_with_source_path,
    )
    process_trace_paths: dict[str, list[Path]] | None = None
    if config.save_process_traces:
        process_trace_paths = save_process_traces(
            model,
            scheduler,
            dataset,
            config,
            device,
        )
        print(f"saved process traces: {config.output_dir / 'process_traces'}")

    result = {
        "dataset_size": len(dataset),
        "num_conditions": dataset.num_conditions,
        "train_steps": effective_train_steps,
        "final_loss": last_loss,
        "checkpoint": final_checkpoint,
        "sample_grid": final_sample_path,
        "sample_source_grid": final_source_path,
        "sample_with_source_grid": final_with_source_path,
        "beta_schedule": beta_schedule_path,
        "train_batches": train_batch_dir,
        "process_traces": (
            None if process_trace_paths is None else config.output_dir / "process_traces"
        ),
    }

    cleanup_state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "dataset": dataset,
        "indexed_dataset": indexed_dataset,
        "dataloader": dataloader,
        "loader_iter": loader_iter,
    }
    model = optimizer = scheduler = dataset = indexed_dataset = dataloader = loader_iter = None
    cleanup_torch_resources(cleanup_state)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a conditional DDPM on generated hash-step images."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ConditionalDiffusionTrainConfig.data_root,
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=ConditionalDiffusionTrainConfig.json_root,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ConditionalDiffusionTrainConfig.output_dir,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=ConditionalDiffusionTrainConfig.image_size,
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3, 4),
        default=ConditionalDiffusionTrainConfig.channels,
    )
    parser.add_argument(
        "--fit-mode",
        choices=("pad", "resize", "reshape", "height-flatten"),
        default=ConditionalDiffusionTrainConfig.fit_mode,
        help=(
            "Image pre-processing mode. "
            "'reshape' flattens pixels and reshapes to an equal-area square."
            " 'height-flatten' keeps the original height and flattens rows into width."
        ),
    )
    parser.add_argument(
        "--condition-mode",
        choices=(
            "json-step",
            "json-value",
            "json-step-value",
            "relative-path",
            "top-level",
            "filename",
        ),
        default=ConditionalDiffusionTrainConfig.condition_mode,
        help=(
            "Retained for backward compatibility. "
            "Training always uses message.png; use --label-source to choose labels."
        ),
    )
    parser.add_argument(
        "--label-source",
        choices=("final-hash",),
        default=ConditionalDiffusionTrainConfig.label_source,
        help="Condition label source. Only final hash labels are supported.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ConditionalDiffusionTrainConfig.batch_size,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=ConditionalDiffusionTrainConfig.train_steps,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=ConditionalDiffusionTrainConfig.epochs,
        help="When set, overrides train_steps with ceil(dataset_size / batch_size) * epochs.",
    )
    parser.add_argument(
        "--timesteps",
        type=_parse_timesteps_arg,
        default=ConditionalDiffusionTrainConfig.timesteps,
        help=(
            "Diffusion timesteps used for linear beta schedule, or 'auto' to "
            "sync linear length to hash approach schedule length. "
            "For file/hash schedules, timesteps follow the beta schedule length."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=ConditionalDiffusionTrainConfig.learning_rate,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=ConditionalDiffusionTrainConfig.base_channels,
    )
    parser.add_argument("--time-dim", type=int, default=ConditionalDiffusionTrainConfig.time_dim)
    parser.add_argument(
        "--beta-start",
        type=float,
        default=ConditionalDiffusionTrainConfig.beta_start,
    )
    parser.add_argument("--beta-end", type=float, default=ConditionalDiffusionTrainConfig.beta_end)
    parser.add_argument(
        "--beta-schedule",
        choices=("linear", "file", "hash-approach1", "hash-approach2"),
        default=ConditionalDiffusionTrainConfig.beta_schedule,
    )
    parser.add_argument(
        "--beta-values-path",
        type=Path,
        default=ConditionalDiffusionTrainConfig.beta_values_path,
    )
    parser.add_argument(
        "--beta-schedule-step",
        default=ConditionalDiffusionTrainConfig.beta_schedule_step,
    )
    parser.add_argument("--device", default=ConditionalDiffusionTrainConfig.device)
    parser.add_argument("--seed", type=int, default=ConditionalDiffusionTrainConfig.seed)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=ConditionalDiffusionTrainConfig.num_workers,
    )
    parser.add_argument("--log-every", type=int, default=ConditionalDiffusionTrainConfig.log_every)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=ConditionalDiffusionTrainConfig.sample_every,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=ConditionalDiffusionTrainConfig.checkpoint_every,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=ConditionalDiffusionTrainConfig.sample_count,
    )
    parser.add_argument(
        "--save-process-traces",
        action="store_true",
        default=ConditionalDiffusionTrainConfig.save_process_traces,
        help="Save forward noising and reverse denoising intermediate PNG grids.",
    )
    parser.add_argument(
        "--trace-sample-count",
        type=int,
        default=ConditionalDiffusionTrainConfig.trace_sample_count,
    )
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=ConditionalDiffusionTrainConfig.trace_steps,
        help=(
            "Deprecated compatibility option. "
            "Reverse and forward process traces are both saved for all timesteps."
        ),
    )
    parser.add_argument(
        "--save-train-batches-every",
        type=int,
        default=ConditionalDiffusionTrainConfig.save_train_batches_every,
        help=(
            "Save actual training input batches as PNG grids every N optimizer steps. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--temporal-conditioning",
        choices=("class", "loop-sinusoidal", "loop-structured", "loop-sequence"),
        default=ConditionalDiffusionTrainConfig.temporal_conditioning,
        help=(
            "Temporal conditioning mode for loop labels. "
            "'class' (default) treats each loop as an independent category. "
            "'loop-sinusoidal' applies sinusoidal PE on the loop index. "
            "'loop-structured' sums sinusoidal PE + start/end boundary projections. "
            "'loop-sequence' runs a Transformer over all loop tokens."
        ),
    )
    parser.add_argument(
        "--use-loop-images",
        action="store_true",
        default=ConditionalDiffusionTrainConfig.use_loop_images,
        help="Load per-loop images using the loop image discovery pipeline.",
    )
    parser.add_argument(
        "--max-loop-count",
        type=int,
        default=ConditionalDiffusionTrainConfig.max_loop_count,
        help="Maximum number of loops (sequence length for loop-sequence mode). Default 64.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ConditionalDiffusionTrainConfig:
    return ConditionalDiffusionTrainConfig(
        data_root=args.data_root,
        json_root=args.json_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        channels=args.channels,
        fit_mode=args.fit_mode,
        condition_mode=args.condition_mode,
        label_source=args.label_source,
        max_images=args.max_images,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        epochs=args.epochs,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        beta_values_path=args.beta_values_path,
        beta_schedule_step=args.beta_schedule_step,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        log_every=args.log_every,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
        sample_count=args.sample_count,
        save_process_traces=args.save_process_traces,
        trace_sample_count=args.trace_sample_count,
        trace_steps=args.trace_steps,
        save_train_batches_every=args.save_train_batches_every,
        temporal_conditioning=args.temporal_conditioning,
        use_loop_images=args.use_loop_images,
        max_loop_count=args.max_loop_count,
    )


def main() -> None:
    result = train_conditional_diffusion(config_from_args(build_arg_parser().parse_args()))
    print(
        "completed "
        f"dataset_size={result['dataset_size']} "
        f"conditions={result['num_conditions']} "
        f"train_steps={result['train_steps']} "
        f"final_loss={result['final_loss']:.6f}"
    )
    print(f"checkpoint: {result['checkpoint']}")
    print(f"sample_grid: {result['sample_grid']}")
    print(f"beta_schedule: {result['beta_schedule']}")
    if result["train_batches"] is not None:
        print(f"train_batches: {result['train_batches']}")
    if result["process_traces"] is not None:
        print(f"process_traces: {result['process_traces']}")


if __name__ == "__main__":
    main()
