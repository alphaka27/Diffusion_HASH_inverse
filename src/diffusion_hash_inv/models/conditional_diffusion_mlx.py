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
from PIL import Image, ImageDraw, ImageOps

import mlx.core as mx


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


FitMode = Literal["resize", "pad"]
LabelSource = Literal["final-hash"]


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
    timesteps: int = 100
    learning_rate: float = 1e-3
    time_dim: int = 64
    hidden_dim: int = 256
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str = "cpu"
    seed: int = 0
    log_every: int = 50
    sample_count: int = 16
    columns: int = 4
    sample_name: str = "samples.png"

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

    converted = image.convert("L" if channels == 1 else "RGB")
    size = (image_size, image_size)
    if fit_mode == "resize":
        return converted.resize(size, Image.Resampling.BICUBIC)
    if fit_mode == "pad":
        color = 0 if channels == 1 else (0, 0, 0)
        return ImageOps.pad(converted, size, method=Image.Resampling.BICUBIC, color=color)
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

    @property
    def image_dim(self) -> int:
        return self.channels * self.image_size * self.image_size

    @property
    def num_conditions(self) -> int:
        return len(self.condition_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: MLXGeneratedImageSample) -> np.ndarray:
        with Image.open(sample.path) as image:
            fitted = _fit_image(image, self.image_size, self.channels, self.fit_mode)
            array = _normalize_image_array(fitted, self.channels)
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
    ) -> None:
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if beta_end < beta_start:
            raise ValueError("beta_end must be greater than or equal to beta_start")

        self.timesteps = int(timesteps)
        self.betas = mx.linspace(beta_start, beta_end, self.timesteps, dtype=mx.float32)
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
    arr = arr.astype(np.uint8)
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
) -> None:
    """Save generated MLX samples as a labeled image grid."""
    mx.eval(samples, labels)
    sample_values = np.asarray(samples)
    label_values = np.asarray(labels).astype(int).tolist()

    rows = math.ceil(sample_values.shape[0] / columns)
    cell = image_size + 14
    mode = "L" if channels == 1 else "RGB"
    background = 255 if channels == 1 else (255, 255, 255)
    grid = Image.new(mode, (columns * image_size, rows * cell), color=background)
    draw = ImageDraw.Draw(grid)

    for idx, vector in enumerate(sample_values):
        row, col = divmod(idx, columns)
        x = col * image_size
        y = row * cell
        grid.paste(_image_from_vector(vector, image_size, channels), (x, y))
        label = label_values[idx]
        text = str(label)
        if idx_to_condition is not None:
            text = idx_to_condition.get(label, text)[:12]
        draw.text((x + 2, y + image_size + 1), text, fill=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def _jsonable_config(config: MLXConditionalDiffusionTrainConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def train_conditional_diffusion_mlx(config: MLXConditionalDiffusionTrainConfig) -> Path:
    """Train a conditional DDPM with MLX and save a sample grid."""
    configure_device(config.device)
    if config.seed is not None:
        mx.random.seed(config.seed)

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
    scheduler = MLXDDPMScheduler(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    train_step = make_train_step(model, scheduler, optimizer)
    mx.eval(model.parameters())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(_jsonable_config(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.output_dir / "label_map.json").write_text(
        json.dumps(dataset.condition_to_idx, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    for step in range(1, config.train_steps + 1):
        x0, labels = dataset.sample_batch(config.batch_size)
        loss = train_step(x0, labels)
        if step == 1 or step % config.log_every == 0 or step == config.train_steps:
            mx.eval(loss)
            print(f"step={step:04d} loss={float(loss):.6f}")

    sample_labels = mx.array(
        [idx % dataset.num_conditions for idx in range(config.sample_count)],
        dtype=mx.int32,
    )
    samples = scheduler.sample(model, sample_labels, dataset.image_dim)
    output_path = config.output_dir / config.sample_name
    save_image_grid(
        samples,
        sample_labels,
        output_path,
        image_size=config.image_size,
        channels=config.channels,
        columns=config.columns,
        idx_to_condition=dataset.idx_to_condition,
    )
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
        choices=("resize", "pad"),
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
        type=int,
        default=MLXConditionalDiffusionTrainConfig.timesteps,
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
        "--sample-count",
        type=int,
        default=MLXConditionalDiffusionTrainConfig.sample_count,
    )
    parser.add_argument("--columns", type=int, default=MLXConditionalDiffusionTrainConfig.columns)
    parser.add_argument("--sample-name", default=MLXConditionalDiffusionTrainConfig.sample_name)
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
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        sample_count=args.sample_count,
        columns=args.columns,
        sample_name=args.sample_name,
    )


def main(argv: Sequence[str] | None = None) -> None:
    output_path = train_conditional_diffusion_mlx(
        config_from_args(build_arg_parser().parse_args(argv))
    )
    print(f"saved samples: {output_path}")


if __name__ == "__main__":
    main()
