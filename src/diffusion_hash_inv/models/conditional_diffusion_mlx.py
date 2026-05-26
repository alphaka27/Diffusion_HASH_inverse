"""
MLX conditional DDPM training pipeline for generated hash images.

This module is the MLX counterpart to the PyTorch conditional DDPM pipeline. It
reads generated ``data/images/<run-id>/message.png`` images, matches each run to
``output/json/**/<run-id>.json``, uses the final hash value as a class label,
and trains a compact MLP denoiser with the DDPM objective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

import mlx.core as mx

from diffusion_hash_inv.analyze import Analyze
from diffusion_hash_inv.config import ImgConfig
from diffusion_hash_inv.models.sample_decoding import write_decode_comparison
from diffusion_hash_inv.scheduling import BetaScheduler


def _preparse_device(argv: list[str]) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--device" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return "cpu"


if __name__ == "__main__":
    configure_target = _preparse_device(sys.argv[1:])
    if configure_target == "cpu":
        mx.set_default_device(mx.cpu)
    elif configure_target == "gpu":
        mx.set_default_device(mx.gpu)

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402


FitMode = Literal["resize", "pad", "height-flatten", "cube-id-grid"]
LabelSource = Literal["final-hash"]
BetaScheduleMode = Literal["linear", "file", "hash-approach1", "hash-approach2"]


@dataclass(frozen=True)
class MLXGeneratedImageSample:
    """One generated image and its integer condition label."""

    path: Path
    run_id: str
    condition: str
    label: int


@dataclass(frozen=True)
class MLXConditionalDiffusionTrainConfig:
    """Runtime configuration for MLX conditional DDPM training."""

    data_root: Path = Path("data/images")
    json_root: Path = Path("output/json")
    output_dir: Path = Path("output/conditional_diffusion_mlx")
    image_size: int = 32
    channels: int = 1
    fit_mode: FitMode = "pad"
    label_source: LabelSource = "final-hash"
    max_images: int | None = None
    batch_size: int = 32
    train_steps: int = 500
    timesteps: int | Literal["auto"] = 100
    learning_rate: float = 1e-3
    time_dim: int = 64
    hidden_dim: int = 256
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: BetaScheduleMode = "linear"
    beta_values_path: Path | None = None
    beta_schedule_step: str = "4th Step"
    device: str = "cpu"
    seed: int = 0
    log_every: int = 50
    sample_every: int = 0
    checkpoint_every: int = 0
    sample_count: int = 16
    save_process_traces: bool = False
    trace_sample_count: int = 4
    columns: int = 4
    sample_name: str = "final.png"
    source_name: str = "source.png"

    @property
    def image_dim(self) -> int:
        return self.channels * self.image_size * self.image_size


def configure_device(device: str) -> None:
    """Select the MLX default device before model parameters are initialized."""
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError(f"Unsupported device: {device}")


def _as_1d_beta_array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if (array <= 0.0).any() or (array >= 1.0).any():
        raise ValueError(f"{name} values must be in the open interval (0, 1)")
    return array


def _load_beta_values(beta_path: Path | str) -> np.ndarray:
    path = Path(beta_path)
    if not path.exists():
        raise FileNotFoundError(f"Beta values file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _as_1d_beta_array(np.load(path), name="betas")
    if suffix == ".npz":
        loaded = np.load(path)
        key = "betas" if "betas" in loaded else loaded.files[0]
        return _as_1d_beta_array(loaded[key], name=f"betas[{key}]")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Beta values file is empty: {path}")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in beta values file: {path} "
                f"({exc.msg} at line {exc.lineno}, column {exc.colno})"
            ) from exc
        values = payload["betas"] if isinstance(payload, dict) and "betas" in payload else payload
        return _as_1d_beta_array(values, name="betas")

    values = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float64)
    return _as_1d_beta_array(values, name="betas")


def _hash_approach_beta_candidates(
    config: MLXConditionalDiffusionTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
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


def build_beta_schedule_mlx(config: MLXConditionalDiffusionTrainConfig) -> np.ndarray | None:
    """Build optional MLX DDPM betas from config."""
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


def _read_json_payload(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"JSON payload file is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON payload in: {path} "
            f"({exc.msg} at line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in: {path}")
    return payload


def _canonical_json_label(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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


def discover_generated_image_samples_mlx(
    root: Path | str,
    *,
    json_root: Path | str = Path("output/json"),
    label_source: LabelSource = "final-hash",
    max_images: int | None = None,
) -> tuple[list[MLXGeneratedImageSample], dict[str, int]]:
    """Discover ``message.png`` files and assign stable final-hash labels."""

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

    json_index = _load_json_index(json_root)
    payload_cache: dict[str, dict[str, object]] = {}
    unlabeled: list[tuple[Path, str, str]] = []
    condition_names: list[str] = []

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
        unlabeled.append((path, run_id, condition))

    condition_to_idx = {name: idx for idx, name in enumerate(sorted(set(condition_names)))}
    samples = [
        MLXGeneratedImageSample(
            path=path,
            run_id=run_id,
            condition=condition,
            label=condition_to_idx[condition],
        )
        for path, run_id, condition in unlabeled
    ]
    return samples, condition_to_idx


def _fit_image(
    image: Image.Image,
    image_size: int,
    channels: int,
    fit_mode: FitMode,
) -> Image.Image:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if fit_mode == "cube-id-grid" and channels != 3:
        raise ValueError("cube-id-grid fit mode requires channels=3 to preserve RGB CubeID values")

    converted = image.convert("L" if channels == 1 else "RGB")
    size = (image_size, image_size)
    if fit_mode == "resize":
        return converted.resize(size, Image.Resampling.BICUBIC)
    if fit_mode == "pad":
        color = 0 if channels == 1 else (0, 0, 0)
        return ImageOps.pad(converted, size, method=Image.Resampling.BICUBIC, color=color)
    if fit_mode in {"height-flatten", "cube-id-grid"}:
        img_width, img_height = ImgConfig().img_size
        if converted.width % img_width != 0 or converted.height % img_height != 0:
            raise ValueError(
                f"{fit_mode} fit mode requires dimensions to be multiples of "
                f"ImgConfig.img_size={ImgConfig().img_size}; "
                f"got {converted.width}x{converted.height}"
            )
        source = np.asarray(converted, dtype=np.uint8)
        if channels == 1:
            source = source[:, :, None]
        rows = converted.height // img_height
        cols = converted.width // img_width
        block_count = rows * cols
        square_blocks = math.isqrt(block_count)
        if square_blocks * square_blocks != block_count:
            raise ValueError(
                f"{fit_mode} fit mode requires the number of ImgConfig-sized blocks "
                f"to be a perfect square (got {block_count})"
            )
        blocks = source.reshape(rows, img_height, cols, img_width, channels).transpose(
            0, 2, 1, 3, 4
        )
        flattened_pixels = blocks[:, :, img_height // 2, img_width // 2, :].reshape(
            block_count, channels
        )
        reshaped = flattened_pixels.reshape(square_blocks, square_blocks, channels)
        if channels == 1:
            return Image.fromarray(reshaped[:, :, 0])
        return Image.fromarray(reshaped)
    raise ValueError(f"Unsupported fit_mode: {fit_mode}")


def _normalize_image_array(image: Image.Image, channels: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if channels == 1:
        if array.ndim == 3:
            array = array[..., 0]
        array = array[None, :, :]
    else:
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        array = np.transpose(array[:, :, :3], (2, 0, 1))
    return array / 127.5 - 1.0


class MLXGeneratedImageDataset:
    """Small MLX-friendly dataset that returns flattened normalized image arrays."""

    def __init__(
        self,
        root: Path | str,
        *,
        json_root: Path | str = Path("output/json"),
        image_size: int = 32,
        channels: int = 1,
        fit_mode: FitMode = "pad",
        label_source: LabelSource = "final-hash",
        max_images: int | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.json_root = Path(json_root)
        self.image_size = image_size
        self.channels = channels
        self.fit_mode = fit_mode
        self.label_source = label_source
        self.samples, self.condition_to_idx = discover_generated_image_samples_mlx(
            self.root,
            json_root=self.json_root,
            label_source=label_source,
            max_images=max_images,
        )
        self.idx_to_condition = {idx: condition for condition, idx in self.condition_to_idx.items()}
        self.rng = np.random.default_rng(seed)
        self.output_image_size = self._infer_output_image_size()

    @property
    def image_dim(self) -> int:
        return self.channels * self.output_image_size * self.output_image_size

    @property
    def num_conditions(self) -> int:
        return len(self.condition_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _infer_output_image_size(self) -> int:
        with Image.open(self.samples[0].path) as image:
            fitted = _fit_image(image, self.image_size, self.channels, self.fit_mode)
        if fitted.width != fitted.height:
            raise ValueError(
                f"MLX fitted images must be square; got {fitted.width}x{fitted.height}"
            )
        return int(fitted.width)

    def _load_image(self, sample: MLXGeneratedImageSample) -> np.ndarray:
        with Image.open(sample.path) as image:
            fitted = _fit_image(image, self.image_size, self.channels, self.fit_mode)
            array = _normalize_image_array(fitted, self.channels)
        expected_shape = (self.channels, self.output_image_size, self.output_image_size)
        if array.shape != expected_shape:
            raise ValueError(
                "All MLX fitted images must have the same shape; "
                f"expected {expected_shape}, got {array.shape} from {sample.path}"
            )
        return array.reshape(-1).astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        sample = self.samples[index]
        return self._load_image(sample), sample.label

    def batch(self, indices: np.ndarray | mx.array) -> tuple[mx.array, mx.array]:
        images: list[np.ndarray] = []
        labels: list[int] = []
        for index in np.asarray(indices).tolist():
            image, label = self[index]
            images.append(image)
            labels.append(label)
        return (
            mx.array(np.stack(images), dtype=mx.float32),
            mx.array(np.asarray(labels, dtype=np.int32), dtype=mx.int32),
        )

    def sample_batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self.rng.integers(0, len(self.samples), size=batch_size)
        return self.batch(indices)


def timestep_embedding(t: mx.array, dim: int, max_period: int = 10_000) -> mx.array:
    """Create sinusoidal embeddings for integer diffusion timesteps."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / max(half - 1, 1))
    args = t.astype(mx.float32)[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


class MLXConditionalDenoiser(nn.Module):
    """Predict Gaussian noise from flattened image vectors, timesteps, and labels."""

    def __init__(
        self,
        image_dim: int,
        num_conditions: int,
        time_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if image_dim <= 0:
            raise ValueError("image_dim must be positive")
        if num_conditions <= 0:
            raise ValueError("num_conditions must be positive")
        if time_dim <= 0:
            raise ValueError("time_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.image_dim = image_dim
        self.time_dim = time_dim
        self.label_embedding = nn.Embedding(num_conditions, time_dim)
        self.fc1 = nn.Linear(image_dim + time_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, image_dim)

    def __call__(self, x: mx.array, t: mx.array, labels: mx.array) -> mx.array:
        t_emb = timestep_embedding(t, self.time_dim)
        y_emb = self.label_embedding(labels)
        h = mx.concatenate([x, t_emb, y_emb], axis=-1)
        h = nn.silu(self.fc1(h))
        h = nn.silu(self.fc2(h))
        return self.fc3(h)


class MLXDDPMScheduler:
    """Forward and reverse diffusion coefficients for MLX DDPM."""

    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        betas: np.ndarray | list[float] | None = None,
    ) -> None:
        if betas is None:
            if timesteps <= 0:
                raise ValueError("timesteps must be positive")
            if beta_end < beta_start:
                raise ValueError("beta_end must be greater than or equal to beta_start")
            self.timesteps = int(timesteps)
            self.betas = mx.linspace(beta_start, beta_end, self.timesteps, dtype=mx.float32)
        else:
            beta_values = _as_1d_beta_array(betas, name="betas")
            self.timesteps = int(beta_values.size)
            self.betas = mx.array(beta_values, dtype=mx.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = mx.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = mx.concatenate(
            [mx.ones((1,), dtype=mx.float32), self.alpha_bars[:-1]],
            axis=0,
        )
        self.sqrt_alpha_bars = mx.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = mx.sqrt(1.0 - self.alpha_bars)
        self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (
            1.0 - self.alpha_bars
        )

    @staticmethod
    def _extract(coefficients: mx.array, t: mx.array, target_ndim: int) -> mx.array:
        values = mx.take(coefficients, t, axis=0)
        return values.reshape((t.shape[0],) + (1,) * (target_ndim - 1))

    def sample_timesteps(self, batch_size: int) -> mx.array:
        return mx.random.randint(0, self.timesteps, shape=(batch_size,), dtype=mx.int32)

    def q_sample(self, x0: mx.array, t: mx.array, noise: mx.array | None = None) -> mx.array:
        if noise is None:
            noise = mx.random.normal(x0.shape, dtype=mx.float32)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.ndim)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.ndim)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def p_sample(
        self,
        model: MLXConditionalDenoiser,
        x: mx.array,
        step: int,
        labels: mx.array,
    ) -> mx.array:
        batch_size = x.shape[0]
        t = mx.full((batch_size,), step, dtype=mx.int32)
        beta_t = self._extract(self.betas, t, x.ndim)
        alpha_t = self._extract(self.alphas, t, x.ndim)
        alpha_bar_t = self._extract(self.alpha_bars, t, x.ndim)
        pred_noise = model(x, t, labels)

        mean = (1.0 / mx.sqrt(alpha_t)) * (
            x - (beta_t / mx.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )
        if step == 0:
            return mean

        variance = self._extract(self.posterior_variance, t, x.ndim)
        return mean + mx.sqrt(variance) * mx.random.normal(x.shape, dtype=mx.float32)

    def sample(
        self,
        model: MLXConditionalDenoiser,
        labels: mx.array,
        image_dim: int,
    ) -> mx.array:
        x = mx.random.normal((labels.shape[0], image_dim), dtype=mx.float32)
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step, labels)
        return mx.clip(x, -1.0, 1.0)

    def sample_with_trace(
        self,
        model: MLXConditionalDenoiser,
        labels: mx.array,
        image_dim: int,
        trace_timesteps: Sequence[int],
    ) -> tuple[mx.array, dict[int, mx.array]]:
        x = mx.random.normal((labels.shape[0], image_dim), dtype=mx.float32)
        trace_set = {int(step) for step in trace_timesteps}
        traces: dict[int, mx.array] = {self.timesteps: mx.clip(x, -1.0, 1.0)}
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step, labels)
            if step in trace_set:
                traces[step] = mx.clip(x, -1.0, 1.0)
        return mx.clip(x, -1.0, 1.0), traces


def diffusion_loss(
    model: MLXConditionalDenoiser,
    scheduler: MLXDDPMScheduler,
    x0: mx.array,
    labels: mx.array,
) -> mx.array:
    t = scheduler.sample_timesteps(x0.shape[0])
    noise = mx.random.normal(x0.shape, dtype=mx.float32)
    xt = scheduler.q_sample(x0, t, noise)
    pred_noise = model(xt, t, labels)
    return mx.mean((pred_noise - noise) ** 2)


def make_train_step(
    model: MLXConditionalDenoiser,
    scheduler: MLXDDPMScheduler,
    optimizer: optim.Optimizer,
):
    loss_and_grad_fn = nn.value_and_grad(
        model,
        lambda m, x0, labels: diffusion_loss(m, scheduler, x0, labels),
    )

    def train_step(x0: mx.array, labels: mx.array) -> mx.array:
        loss, grads = loss_and_grad_fn(model, x0, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    return train_step


def _image_from_vector(vector: np.ndarray, image_size: int, channels: int) -> Image.Image:
    arr = ((vector.reshape(channels, image_size, image_size) + 1.0) * 127.5).clip(0, 255)
    arr = np.rint(arr).astype(np.uint8)
    if channels == 1:
        return Image.fromarray(arr[0])
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


def save_image_grid(
    samples: mx.array,
    labels: mx.array,
    output_path: Path,
    *,
    image_size: int,
    channels: int,
    columns: int = 4,
    idx_to_condition: dict[int, str] | None = None,
    json_path: Path | None = None,
    single_file: bool = True,
) -> dict[str, Path | list[Path] | None]:
    """Save generated MLX samples as a grid or as one PNG per sample."""
    mx.eval(samples, labels)
    sample_values = np.asarray(samples)
    label_values = np.asarray(labels).astype(int).tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    saved_paths: list[Path] = []
    if sample_values.shape[0] == 1:
        _image_from_vector(sample_values[0], image_size, channels).save(output_path)
        saved_files.append(output_path.name)
        saved_paths.append(output_path)
        columns_for_labels = 1
    elif single_file:
        rows = math.ceil(sample_values.shape[0] / columns)
        mode = "L" if channels == 1 else "RGB"
        background = 255 if channels == 1 else (255, 255, 255)
        grid = Image.new(mode, (columns * image_size, rows * image_size), color=background)

        for idx, vector in enumerate(sample_values):
            row, col = divmod(idx, columns)
            x = col * image_size
            y = row * image_size
            grid.paste(_image_from_vector(vector, image_size, channels), (x, y))

        grid.save(output_path)
        saved_files = [output_path.name for _ in range(sample_values.shape[0])]
        saved_paths = [output_path]
        columns_for_labels = columns
    else:
        stem = output_path.stem
        suffix = output_path.suffix
        for idx, vector in enumerate(sample_values):
            file_path = output_path.with_name(f"{stem}_{idx:03d}{suffix}")
            _image_from_vector(vector, image_size, channels).save(file_path)
            saved_files.append(file_path.name)
            saved_paths.append(file_path)
        columns_for_labels = 1

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                [
                    {
                        "index": idx,
                        "file": saved_files[idx],
                        "label": int(label),
                        "condition": (
                            idx_to_condition.get(int(label), str(label))
                            if idx_to_condition is not None
                            else str(label)
                        ),
                        **(
                            {
                                "row": idx // columns_for_labels,
                                "column": idx % columns_for_labels,
                            }
                            if single_file and sample_values.shape[0] > 1
                            else {}
                        ),
                    }
                    for idx, label in enumerate(label_values)
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    return {"manifest": json_path, "files": saved_paths}


def save_sample_artifacts_mlx(
    source_samples: mx.array,
    generated_samples: mx.array,
    labels: mx.array,
    output_dir: Path,
    *,
    image_size: int,
    channels: int,
    columns: int,
    idx_to_condition: dict[int, str],
    source_name: str = "source.png",
    final_name: str = "final.png",
    fit_mode: str = "pad",
) -> dict[str, Path | list[Path]]:
    mx.eval(source_samples, generated_samples)
    if source_samples.shape != generated_samples.shape:
        raise ValueError(
            "source_samples and generated_samples must have the same shape, "
            f"got {source_samples.shape} and {generated_samples.shape}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source"
    final_dir = output_dir / "final"
    source_path = source_dir / source_name
    final_path = final_dir / final_name
    source_manifest = source_path.with_suffix(".labels.json")
    final_manifest = final_path.with_suffix(".labels.json")
    stale_paths = [output_dir / "preview.png", output_dir / "preview.labels.json"]
    stale_paths.extend([output_dir / source_name, output_dir / final_name])
    stale_paths.extend(output_dir.glob(f"{source_path.stem}_*{source_path.suffix}"))
    stale_paths.extend(output_dir.glob(f"{final_path.stem}_*{final_path.suffix}"))
    stale_paths.extend(source_dir.glob(f"{source_path.stem}_*{source_path.suffix}"))
    stale_paths.extend(final_dir.glob(f"{final_path.stem}_*{final_path.suffix}"))
    if source_samples.shape[0] > 1:
        stale_paths.extend([source_path, final_path])
    for stale_path in stale_paths:
        if stale_path.is_file():
            stale_path.unlink()
    source_result = save_image_grid(
        source_samples,
        labels,
        source_path,
        image_size=image_size,
        channels=channels,
        columns=columns,
        idx_to_condition=idx_to_condition,
        json_path=source_manifest,
        single_file=False,
    )
    final_result = save_image_grid(
        generated_samples,
        labels,
        final_path,
        image_size=image_size,
        channels=channels,
        columns=columns,
        idx_to_condition=idx_to_condition,
        json_path=final_manifest,
        single_file=False,
    )
    decode_comparison = write_decode_comparison(
        source_result["files"],
        final_result["files"],
        output_dir / "decode_comparison.json",
        fit_mode=fit_mode,
    )
    return {
        "source": source_manifest,
        "source_files": source_result["files"],
        "final": final_manifest,
        "final_files": final_result["files"],
        "source_dir": source_dir,
        "final_dir": final_dir,
        "decode_comparison": decode_comparison,
    }


def _reference_vectors_for_labels(
    dataset: MLXGeneratedImageDataset,
    labels: mx.array,
) -> mx.array:
    first_index_by_label: dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        first_index_by_label.setdefault(int(sample.label), index)

    mx.eval(labels)
    vectors: list[np.ndarray] = []
    for label in np.asarray(labels).astype(int).tolist():
        if label not in first_index_by_label:
            raise ValueError(f"Label {label} not found in dataset")
        vector, _ = dataset[first_index_by_label[label]]
        vectors.append(vector)
    return mx.array(np.stack(vectors), dtype=mx.float32)


def _save_forward_process_trace_mlx(
    scheduler: MLXDDPMScheduler,
    images: mx.array,
    labels: mx.array,
    dataset: MLXGeneratedImageDataset,
    output_dir: Path,
    columns: int,
) -> list[Path]:
    png_dir = output_dir / "png"
    json_dir = output_dir / "json"
    png_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    print(f"[forward-trace] saving x0 + {scheduler.timesteps} noising steps to {output_dir}")

    x0_path = png_dir / "x0.png"
    save_image_grid(
        images,
        labels,
        x0_path,
        image_size=dataset.output_image_size,
        channels=dataset.channels,
        columns=columns,
        idx_to_condition=dataset.idx_to_condition,
        json_path=json_dir / "x0.labels.json",
    )
    saved_paths.append(x0_path)
    print(f"[forward-trace] saved x0: {x0_path}")

    noise = mx.random.normal(images.shape, dtype=mx.float32)
    for step in range(scheduler.timesteps):
        timesteps = mx.full((images.shape[0],), step, dtype=mx.int32)
        noised = scheduler.q_sample(images, timesteps, noise)
        path = png_dir / f"t_{step:06d}.png"
        save_image_grid(
            noised,
            labels,
            path,
            image_size=dataset.output_image_size,
            channels=dataset.channels,
            columns=columns,
            idx_to_condition=dataset.idx_to_condition,
            json_path=json_dir / f"t_{step:06d}.labels.json",
        )
        saved_paths.append(path)
        print(f"[forward-trace] step={step:06d} path={path}")

    print(f"[forward-trace] completed: {len(saved_paths)} files")
    return saved_paths


def _save_reverse_process_trace_mlx(
    scheduler: MLXDDPMScheduler,
    model: MLXConditionalDenoiser,
    labels: mx.array,
    dataset: MLXGeneratedImageDataset,
    output_dir: Path,
    columns: int,
) -> list[Path]:
    png_dir = output_dir / "png"
    json_dir = output_dir / "json"
    png_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    print(f"[reverse-trace] saving xT + {scheduler.timesteps} denoising steps to {output_dir}")
    _, traces = scheduler.sample_with_trace(
        model,
        labels,
        dataset.image_dim,
        range(scheduler.timesteps),
    )

    for step in sorted(traces.keys(), reverse=True):
        filename = "xT_noise.png" if step == scheduler.timesteps else f"t_{step:06d}.png"
        path = png_dir / filename
        json_name = filename.replace(".png", ".labels.json")
        save_image_grid(
            traces[step],
            labels,
            path,
            image_size=dataset.output_image_size,
            channels=dataset.channels,
            columns=columns,
            idx_to_condition=dataset.idx_to_condition,
            json_path=json_dir / json_name,
        )
        saved_paths.append(path)
        if step == scheduler.timesteps:
            print(f"[reverse-trace] saved xT: {path}")
        else:
            print(f"[reverse-trace] step={step:06d} path={path}")
    print(f"[reverse-trace] completed: {len(saved_paths)} files")
    return saved_paths


def save_process_traces_mlx(
    model: MLXConditionalDenoiser,
    scheduler: MLXDDPMScheduler,
    dataset: MLXGeneratedImageDataset,
    config: MLXConditionalDiffusionTrainConfig,
) -> dict[str, list[Path]]:
    if config.trace_sample_count <= 0:
        raise ValueError("trace_sample_count must be positive")

    sample_count = min(config.trace_sample_count, len(dataset))
    vectors = []
    labels = []
    for index in range(sample_count):
        vector, label = dataset[index]
        vectors.append(vector)
        labels.append(int(label))
    image_batch = mx.array(np.stack(vectors), dtype=mx.float32)
    label_batch = mx.array(labels, dtype=mx.int32)
    trace_dir = config.output_dir / "process_traces"

    forward_paths = _save_forward_process_trace_mlx(
        scheduler,
        image_batch,
        label_batch,
        dataset,
        trace_dir / "forward",
        config.columns,
    )
    reverse_paths = _save_reverse_process_trace_mlx(
        scheduler,
        model,
        label_batch,
        dataset,
        trace_dir / "reverse",
        config.columns,
    )
    return {"forward": forward_paths, "reverse": reverse_paths}


def _jsonable_config(config: MLXConditionalDiffusionTrainConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def save_beta_schedule_mlx(
    scheduler: MLXDDPMScheduler,
    config: MLXConditionalDiffusionTrainConfig,
) -> Path:
    """Save the concrete beta schedule used by MLX training."""
    path = config.output_dir / "beta_schedule.json"
    mx.eval(scheduler.betas)
    betas = np.asarray(scheduler.betas, dtype=np.float64)
    payload = {
        "mode": config.beta_schedule,
        "timesteps": scheduler.timesteps,
        "beta_start": float(betas[0]),
        "beta_end": float(betas[-1]),
        "beta_schedule_step": config.beta_schedule_step,
        "betas": [float(value) for value in betas.tolist()],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_checkpoint_mlx(
    model: MLXConditionalDenoiser,
    optimizer: optim.Optimizer,
    step: int,
    loss: float,
    dataset: MLXGeneratedImageDataset,
    config: MLXConditionalDiffusionTrainConfig,
) -> Path:
    """Save MLX model weights, optimizer state, and metadata for one step."""
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"step_{step:06d}"
    model_weights_path = checkpoint_dir / f"{prefix}.safetensors"
    optimizer_state_path = checkpoint_dir / f"{prefix}.optimizer.safetensors"
    metadata_path = checkpoint_dir / f"{prefix}.json"

    mx.eval(model.parameters(), optimizer.state)
    model.save_weights(str(model_weights_path))
    mx.save_safetensors(
        str(optimizer_state_path),
        dict(tree_flatten(optimizer.state, destination={})),
    )

    metadata = {
        "step": step,
        "loss": loss if math.isfinite(loss) else None,
        "model_weights": model_weights_path.name,
        "optimizer_state": optimizer_state_path.name,
        "condition_to_idx": dataset.condition_to_idx,
        "config": _jsonable_config(config),
        "model_args": {
            "image_dim": dataset.image_dim,
            "num_conditions": dataset.num_conditions,
            "time_dim": config.time_dim,
            "hidden_dim": config.hidden_dim,
        },
        "scheduler_args": {
            "timesteps": config.timesteps,
            "beta_schedule": config.beta_schedule,
            "beta_schedule_step": config.beta_schedule_step,
            "beta_start": config.beta_start,
            "beta_end": config.beta_end,
        },
        "optimizer_args": {
            "name": "Adam",
            "learning_rate": config.learning_rate,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def train_conditional_diffusion_mlx(config: MLXConditionalDiffusionTrainConfig) -> Path:
    """Train a conditional DDPM with MLX and save a sample grid."""
    configure_device(config.device)
    if config.seed is not None:
        mx.random.seed(config.seed)
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")

    dataset = MLXGeneratedImageDataset(
        config.data_root,
        json_root=config.json_root,
        image_size=config.image_size,
        channels=config.channels,
        fit_mode=config.fit_mode,
        label_source=config.label_source,
        max_images=config.max_images,
        seed=config.seed,
    )
    model = MLXConditionalDenoiser(
        image_dim=dataset.image_dim,
        num_conditions=dataset.num_conditions,
        time_dim=config.time_dim,
        hidden_dim=config.hidden_dim,
    )
    custom_betas = build_beta_schedule_mlx(config)
    scheduler_timesteps = (
        int(config.timesteps)
        if isinstance(config.timesteps, int)
        else int(custom_betas.size if custom_betas is not None else 0)
    )
    if scheduler_timesteps <= 0:
        raise ValueError("timesteps must be positive or resolvable with timesteps='auto'")
    scheduler = MLXDDPMScheduler(
        timesteps=scheduler_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        betas=custom_betas,
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    train_step = make_train_step(model, scheduler, optimizer)
    mx.eval(model.parameters())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(_jsonable_config(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    beta_schedule_path = save_beta_schedule_mlx(scheduler, config)
    (config.output_dir / "label_map.json").write_text(
        json.dumps(dataset.condition_to_idx, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"saved beta schedule: {beta_schedule_path}")

    last_loss = math.nan
    last_checkpoint_step = 0
    sample_dir = config.output_dir / "sample"
    for step in range(1, config.train_steps + 1):
        x0, labels = dataset.sample_batch(config.batch_size)
        loss = train_step(x0, labels)
        mx.eval(loss)
        last_loss = float(loss)
        if step == 1 or step % config.log_every == 0 or step == config.train_steps:
            print(f"step={step:04d} loss={last_loss:.6f}")
        if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
            checkpoint_path = save_checkpoint_mlx(
                model,
                optimizer,
                step,
                last_loss,
                dataset,
                config,
            )
            last_checkpoint_step = step
            print(f"saved checkpoint: {checkpoint_path}")
        if config.sample_every > 0 and step % config.sample_every == 0:
            sample_labels = mx.array(
                [idx % dataset.num_conditions for idx in range(config.sample_count)],
                dtype=mx.int32,
            )
            samples = scheduler.sample(model, sample_labels, dataset.image_dim)
            source_samples = _reference_vectors_for_labels(dataset, sample_labels)
            sample_paths = save_sample_artifacts_mlx(
                source_samples,
                samples,
                sample_labels,
                sample_dir / f"step_{step:06d}",
                image_size=dataset.output_image_size,
                channels=config.channels,
                columns=config.columns,
                idx_to_condition=dataset.idx_to_condition,
                fit_mode=config.fit_mode,
            )
            print(f"saved sample source manifest: {sample_paths['source']}")
            print(f"saved sample final manifest: {sample_paths['final']}")
            print(f"saved sample decode comparison: {sample_paths['decode_comparison']}")

    if last_checkpoint_step != config.train_steps:
        final_checkpoint_path = save_checkpoint_mlx(
            model,
            optimizer,
            config.train_steps,
            last_loss,
            dataset,
            config,
        )
        print(f"saved final checkpoint: {final_checkpoint_path}")

    sample_labels = mx.array(
        [idx % dataset.num_conditions for idx in range(config.sample_count)],
        dtype=mx.int32,
    )
    samples = scheduler.sample(model, sample_labels, dataset.image_dim)
    source_samples = _reference_vectors_for_labels(dataset, sample_labels)
    final_sample_paths = save_sample_artifacts_mlx(
        source_samples,
        samples,
        sample_labels,
        sample_dir,
        image_size=dataset.output_image_size,
        channels=config.channels,
        columns=config.columns,
        idx_to_condition=dataset.idx_to_condition,
        source_name=config.source_name,
        final_name=config.sample_name,
        fit_mode=config.fit_mode,
    )
    output_path = final_sample_paths["final"]
    if config.save_process_traces:
        save_process_traces_mlx(model, scheduler, dataset, config)
        print(f"saved process traces: {config.output_dir / 'process_traces'}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an MLX conditional DDPM on generated hash images."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=MLXConditionalDiffusionTrainConfig.data_root,
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=MLXConditionalDiffusionTrainConfig.json_root,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MLXConditionalDiffusionTrainConfig.output_dir,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.image_size,
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3),
        default=MLXConditionalDiffusionTrainConfig.channels,
    )
    parser.add_argument(
        "--fit-mode",
        choices=("resize", "pad", "height-flatten", "cube-id-grid"),
        default=MLXConditionalDiffusionTrainConfig.fit_mode,
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.batch_size,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.train_steps,
    )
    parser.add_argument(
        "--timesteps",
        type=_parse_timesteps_arg,
        default=MLXConditionalDiffusionTrainConfig.timesteps,
        help=(
            "Diffusion timesteps for linear beta schedule, or 'auto' to sync "
            "linear length to hash approach schedule length. File/hash schedules "
            "use the beta schedule length."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=MLXConditionalDiffusionTrainConfig.learning_rate,
    )
    parser.add_argument("--time-dim", type=int, default=MLXConditionalDiffusionTrainConfig.time_dim)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.hidden_dim,
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=MLXConditionalDiffusionTrainConfig.beta_start,
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=MLXConditionalDiffusionTrainConfig.beta_end,
    )
    parser.add_argument(
        "--beta-schedule",
        choices=("linear", "file", "hash-approach1", "hash-approach2"),
        default=MLXConditionalDiffusionTrainConfig.beta_schedule,
    )
    parser.add_argument(
        "--beta-values-path",
        type=Path,
        default=MLXConditionalDiffusionTrainConfig.beta_values_path,
    )
    parser.add_argument(
        "--beta-schedule-step",
        default=MLXConditionalDiffusionTrainConfig.beta_schedule_step,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=MLXConditionalDiffusionTrainConfig.device,
    )
    parser.add_argument("--seed", type=int, default=MLXConditionalDiffusionTrainConfig.seed)
    parser.add_argument(
        "--log-every",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.log_every,
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.sample_every,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.checkpoint_every,
        help="Save an intermediate checkpoint every N optimizer steps; final is always saved.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.sample_count,
    )
    parser.add_argument(
        "--save-process-traces",
        action="store_true",
        default=MLXConditionalDiffusionTrainConfig.save_process_traces,
        help="Save forward noising and reverse denoising PNG grids for all timesteps.",
    )
    parser.add_argument(
        "--trace-sample-count",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.trace_sample_count,
    )
    parser.add_argument("--columns", type=int, default=MLXConditionalDiffusionTrainConfig.columns)
    parser.add_argument("--sample-name", default=MLXConditionalDiffusionTrainConfig.sample_name)
    parser.add_argument("--source-name", default=MLXConditionalDiffusionTrainConfig.source_name)
    return parser


def config_from_args(args: argparse.Namespace) -> MLXConditionalDiffusionTrainConfig:
    return MLXConditionalDiffusionTrainConfig(
        data_root=args.data_root,
        json_root=args.json_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        channels=args.channels,
        fit_mode=args.fit_mode,
        max_images=args.max_images,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        beta_values_path=args.beta_values_path,
        beta_schedule_step=args.beta_schedule_step,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
        sample_count=args.sample_count,
        save_process_traces=args.save_process_traces,
        trace_sample_count=args.trace_sample_count,
        columns=args.columns,
        sample_name=args.sample_name,
        source_name=args.source_name,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = config_from_args(build_arg_parser().parse_args(argv))
    output_path = train_conditional_diffusion_mlx(config)
    sample_dir = output_path.parent.parent
    source_manifest = (sample_dir / "source" / config.source_name).with_suffix(".labels.json")
    print(f"saved sample dir: {sample_dir}")
    print(f"saved source manifest: {source_manifest}")
    print(f"saved final manifest: {output_path}")
    print(f"saved decode comparison: {sample_dir / 'decode_comparison.json'}")


if __name__ == "__main__":
    main()
