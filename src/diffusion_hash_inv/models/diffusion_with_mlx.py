
"""
Conditional DDPM toy example implemented with MLX.

The script trains a small MLP denoiser on synthetic class-conditioned image
prototypes, then samples new images conditioned on labels. It is intentionally
self-contained so it can be run before wiring in a real dataset.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mlx.core as mx


def _preparse_device(argv: list[str]) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--device" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return "cpu"


if __name__ == "__main__":
    # Importing mlx.nn can compile activation helpers on the default device, so
    # choose the CLI device before importing the neural-network module.
    configure_target = _preparse_device(sys.argv[1:])
    if configure_target == "cpu":
        mx.set_default_device(mx.cpu)
    elif configure_target == "gpu":
        mx.set_default_device(mx.gpu)

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402


@dataclass(frozen=True)
class DiffusionConfig:
    image_size: int = 16
    num_classes: int = 10
    timesteps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    time_dim: int = 64
    hidden_dim: int = 256

    @property
    def image_dim(self) -> int:
        return self.image_size * self.image_size


def configure_device(device: str) -> None:
    """Select the MLX default device before model parameters are initialized."""
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError(f"Unsupported device: {device}")


def timestep_embedding(t: mx.array, dim: int, max_period: int = 10_000) -> mx.array:
    """Create sinusoidal embeddings for integer diffusion timesteps."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / max(half - 1, 1))
    args = t.astype(mx.float32)[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


class ConditionalDenoiser(nn.Module):
    """Predicts Gaussian noise from a noised vector, timestep, and class label."""

    def __init__(
        self,
        image_dim: int,
        num_classes: int,
        time_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.label_embedding = nn.Embedding(num_classes, time_dim)
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


class DDPMScheduler:
    """Forward and reverse diffusion utilities for a linear-beta DDPM."""

    def __init__(
        self,
        timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.timesteps = timesteps
        self.betas = mx.linspace(beta_start, beta_end, timesteps, dtype=mx.float32)
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
        model: ConditionalDenoiser,
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
        model: ConditionalDenoiser,
        labels: mx.array,
        image_dim: int,
    ) -> mx.array:
        x = mx.random.normal((labels.shape[0], image_dim), dtype=mx.float32)
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step, labels)
        return mx.clip(x, -1.0, 1.0)


def make_class_prototypes(num_classes: int, image_size: int) -> mx.array:
    """Build deterministic label-conditioned prototype images in [-1, 1]."""
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    center = (image_size - 1) / 2.0
    radius = image_size * 0.28
    sigma = max(image_size * 0.09, 1.0)
    prototypes: list[np.ndarray] = []

    for label in range(num_classes):
        angle = 2.0 * math.pi * label / num_classes
        cx = center + math.cos(angle) * radius
        cy = center + math.sin(angle) * radius
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
        stripe = 0.25 * (np.cos(xx * math.cos(angle) + yy * math.sin(angle)) + 1.0)
        image = blob + stripe
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        prototypes.append(image * 2.0 - 1.0)

    stacked = np.stack(prototypes).reshape(num_classes, image_size * image_size)
    return mx.array(stacked, dtype=mx.float32)


def synthetic_batch(
    prototypes: mx.array,
    batch_size: int,
    noise_std: float = 0.05,
) -> tuple[mx.array, mx.array]:
    labels = mx.random.randint(0, prototypes.shape[0], shape=(batch_size,), dtype=mx.int32)
    x0 = mx.take(prototypes, labels, axis=0)
    x0 = x0 + noise_std * mx.random.normal(x0.shape, dtype=mx.float32)
    return mx.clip(x0, -1.0, 1.0), labels


def diffusion_loss(
    model: ConditionalDenoiser,
    scheduler: DDPMScheduler,
    x0: mx.array,
    labels: mx.array,
) -> mx.array:
    t = scheduler.sample_timesteps(x0.shape[0])
    noise = mx.random.normal(x0.shape, dtype=mx.float32)
    xt = scheduler.q_sample(x0, t, noise)
    pred_noise = model(xt, t, labels)
    return mx.mean((pred_noise - noise) ** 2)


def make_train_step(
    model: ConditionalDenoiser,
    scheduler: DDPMScheduler,
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


def save_image_grid(
    samples: mx.array,
    labels: mx.array,
    output_path: Path,
    image_size: int,
    columns: int = 5,
) -> None:
    samples = ((samples.reshape(samples.shape[0], image_size, image_size) + 1.0) * 127.5).astype(
        mx.uint8
    )
    mx.eval(samples, labels)
    images = np.array(samples)
    label_values = np.array(labels).tolist()

    rows = math.ceil(images.shape[0] / columns)
    cell = image_size + 14
    grid = Image.new("L", (columns * image_size, rows * cell), color=255)
    draw = ImageDraw.Draw(grid)

    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        x = col * image_size
        y = row * cell
        grid.paste(Image.fromarray(image), (x, y))
        draw.text((x + 2, y + image_size + 1), str(label_values[idx]), fill=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def run_demo(args: argparse.Namespace) -> Path:
    configure_device(args.device)
    if args.seed is not None:
        mx.random.seed(args.seed)

    config = DiffusionConfig(
        image_size=args.image_size,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
    )
    prototypes = make_class_prototypes(config.num_classes, config.image_size)
    model = ConditionalDenoiser(
        image_dim=config.image_dim,
        num_classes=config.num_classes,
        time_dim=config.time_dim,
        hidden_dim=config.hidden_dim,
    )
    scheduler = DDPMScheduler(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    train_step = make_train_step(model, scheduler, optimizer)
    mx.eval(model.parameters(), prototypes)

    for step in range(1, args.train_steps + 1):
        x0, labels = synthetic_batch(prototypes, args.batch_size)
        loss = train_step(x0, labels)
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            mx.eval(loss)
            print(f"step={step:04d} loss={float(loss):.6f}")

    sample_labels = mx.array(
        [idx % config.num_classes for idx in range(args.samples)],
        dtype=mx.int32,
    )
    samples = scheduler.sample(model, sample_labels, config.image_dim)
    save_image_grid(samples, sample_labels, args.output, config.image_size, args.columns)
    return args.output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLX conditional diffusion toy example")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/conditional_diffusion_mlx_samples.png"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    output_path = run_demo(build_arg_parser().parse_args(argv))
    print(f"saved samples: {output_path}")


if __name__ == "__main__":
    main()
