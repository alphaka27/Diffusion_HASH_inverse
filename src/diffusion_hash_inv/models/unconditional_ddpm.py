"""
Unconditional DDPM training pipeline for generated hash images.

This module trains on ``data/images/<run-id>/message.png`` files without using
JSON labels or any other condition. The denoiser has the standard
``epsilon_theta(x_t, t)`` interface.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_hash_inv.models.conditional_diffusion import (
    BetaScheduleMode,
    ConditionalResBlock,
    DDPMNoiseScheduler,
    FitMode,
    SinusoidalTimeEmbedding,
    _denormalize_images,
    _ensure_square_batch,
    _fit_image,
    _group_count,
    _image_from_tensor,
    _normalize_image_array,
    _parse_timesteps_arg,
    _save_forward_process_trace,
    build_beta_schedule,
    resolve_device,
    resolve_train_steps,
    save_beta_schedule,
    save_image_grid,
    set_seed,
)


@dataclass(frozen=True)
class UnconditionalImageSample:
    """One generated image used for unconditional training."""

    path: Path
    run_id: str


@dataclass(frozen=True)
class UnconditionalDDPMTrainConfig:
    """Runtime configuration for unconditional DDPM training."""

    data_root: Path = Path("data/images")
    json_root: Path = Path("output/json")
    output_dir: Path = Path("output/unconditional_ddpm")
    image_size: int = 64
    channels: int = 3
    fit_mode: FitMode = "reshape"
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
    save_train_batches_every: int = 0
    save_process_traces: bool = False
    trace_sample_count: int = 4


def discover_unconditional_image_samples(
    root: Path | str,
    *,
    max_images: int | None = None,
) -> list[UnconditionalImageSample]:
    """Discover ``message.png`` files under ``data/images/<run-id>``."""

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

    samples: list[UnconditionalImageSample] = []
    for path in image_paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(
                "message.png images must be stored under data/images/<run-id>/message.png"
            )
        samples.append(UnconditionalImageSample(path=path, run_id=relative.parts[0]))
    return samples


class UnconditionalImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset that returns ``(image, sample_index)`` without labels."""

    def __init__(
        self,
        root: Path | str,
        image_size: int = 64,
        channels: int = 3,
        fit_mode: FitMode = "reshape",
        max_images: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.fit_mode = fit_mode
        self.source_channels = channels
        self.channels = channels
        self.samples = discover_unconditional_image_samples(
            self.root,
            max_images=max_images,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            fitted = _fit_image(image, self.image_size, self.fit_mode, self.source_channels)
            array = _normalize_image_array(fitted, self.channels)
        image_tensor = torch.from_numpy(array).to(dtype=torch.float32)
        return image_tensor, torch.tensor(index, dtype=torch.long)


class UnconditionalUNet(nn.Module):
    """Small U-Net that predicts DDPM noise from image and timestep only."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

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

    @staticmethod
    def _match_spatial(x: Tensor, target: Tensor) -> Tensor:
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        emb = self.time_embedding(timesteps)
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


class UnconditionalDDPMScheduler(DDPMNoiseScheduler):
    """DDPM scheduler with label-free reverse sampling."""

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: Tensor, step: int) -> Tensor:
        timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        beta_t = self._extract(self.betas, timesteps, x.ndim)
        alpha_t = self._extract(self.alphas, timesteps, x.ndim)
        alpha_bar_t = self._extract(self.alpha_bars, timesteps, x.ndim)
        pred_noise = model(x, timesteps)

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
        device: torch.device | str,
    ) -> Tensor:
        x = torch.randn(shape, device=torch.device(device))
        model.eval()
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step)
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_with_trace(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device | str,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """Run full reverse diffusion, returning all per-step intermediate images."""
        x = torch.randn(shape, device=torch.device(device))
        model.eval()
        traces: dict[int, Tensor] = {self.timesteps: x.clamp(-1.0, 1.0).detach().cpu()}
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step)
            traces[step] = x.clamp(-1.0, 1.0).detach().cpu()
        return x.clamp(-1.0, 1.0), traces


def _print_preprocess_summary(dataset: UnconditionalImageDataset, fit_mode: FitMode) -> None:
    sample_path = dataset.samples[0].path
    with Image.open(sample_path) as sample_image:
        src_w, src_h = sample_image.width, sample_image.height
    out_c, out_h, out_w = (int(value) for value in dataset[0][0].shape)
    print(
        f"[reshape] mode={fit_mode} source={src_w}x{src_h} "
        f"output={out_w}x{out_h} channels={out_c}"
    )


def save_unconditional_images(
    images: Tensor,
    sample_indices: Tensor | None,
    dataset: UnconditionalImageDataset | None,
    path: Path,
) -> Path:
    """Save generated images with optional source sample metadata."""

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

    metadata: list[dict[str, object]] = []
    indices = [] if sample_indices is None else sample_indices.detach().cpu().tolist()
    for idx, filename in enumerate(saved_files):
        record: dict[str, object] = {"index": idx, "file": filename}
        if dataset is not None and idx < len(indices):
            sample = dataset.samples[int(indices[idx])]
            record.update(
                {
                    "dataset_index": int(indices[idx]),
                    "run_id": sample.run_id,
                    "source_path": str(sample.path),
                }
            )
        metadata.append(record)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return path


def _sample_source_indices(
    dataset: UnconditionalImageDataset,
    sample_count: int,
    device: torch.device,
) -> Tensor:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    indices = [index for _, index in zip(range(sample_count), cycle(range(len(dataset))))]
    return torch.tensor(indices, dtype=torch.long, device=device)


def _checkpoint_payload(
    model: UnconditionalUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: UnconditionalDDPMTrainConfig,
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
        "config": config_dict,
        "model_args": {
            "in_channels": model.in_channels,
            "base_channels": model.base_channels,
            "time_dim": model.time_dim,
        },
    }


def save_checkpoint(
    model: UnconditionalUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: UnconditionalDDPMTrainConfig,
) -> Path:
    path = config.output_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, step, loss, config), path)
    return path


def save_train_batch_grid(
    images: Tensor,
    indices: Tensor,
    dataset: UnconditionalImageDataset,
    output_dir: Path,
    step: int,
) -> Path:
    path = output_dir / f"step_{step:06d}.png"
    return save_unconditional_images(images, indices, dataset, path)


def save_unconditional_process_traces(
    model: nn.Module,
    scheduler: UnconditionalDDPMScheduler,
    dataset: UnconditionalImageDataset,
    config: UnconditionalDDPMTrainConfig,
    device: torch.device,
) -> dict[str, list[Path]]:
    """Save forward noising and unconditional reverse denoising traces."""
    if config.trace_sample_count <= 0:
        raise ValueError("trace_sample_count must be positive")

    sample_count = min(config.trace_sample_count, len(dataset))
    images = []
    for index in range(sample_count):
        image, _ = dataset[index]
        images.append(image)

    image_batch = torch.stack(images).to(device)
    label_batch = torch.arange(sample_count, dtype=torch.long, device=device)
    condition_names = [f"sample_{i}" for i in range(sample_count)]
    trace_dir = config.output_dir / "process_traces"
    sample_shape = (image_batch.shape[0], *image_batch.shape[1:])

    forward_paths = _save_forward_process_trace(
        scheduler,
        image_batch,
        label_batch,
        condition_names,
        trace_dir / "forward",
    )

    reverse_dir = trace_dir / "reverse"
    reverse_png_dir = reverse_dir / "png"
    reverse_json_dir = reverse_dir / "json"
    reverse_png_dir.mkdir(parents=True, exist_ok=True)
    reverse_json_dir.mkdir(parents=True, exist_ok=True)
    print(f"[reverse-trace] saving xT + {scheduler.timesteps} denoising steps to {reverse_dir}")
    _, traces = scheduler.sample_with_trace(model, sample_shape, device)
    saved_paths: list[Path] = []
    for step in sorted(traces.keys(), reverse=True):
        filename = "xT_noise.png" if step == scheduler.timesteps else f"t_{step:06d}.png"
        path = reverse_png_dir / filename
        json_name = filename.replace(".png", ".labels.json")
        save_image_grid(
            traces[step], label_batch.detach().cpu(), condition_names, path,
            json_path=reverse_json_dir / json_name,
        )
        saved_paths.append(path)
        if step == scheduler.timesteps:
            print(f"[reverse-trace] saved xT: {path}")
        else:
            print(f"[reverse-trace] step={step:06d} path={path}")
    print(f"[reverse-trace] completed: {len(saved_paths)} files")

    return {"forward": forward_paths, "reverse": saved_paths}


def train_unconditional_ddpm(
    config: UnconditionalDDPMTrainConfig,
) -> dict[str, Path | float | int | None]:
    """Train an unconditional DDPM on generated ``message.png`` images."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = UnconditionalImageDataset(
        root=config.data_root,
        image_size=config.image_size,
        channels=config.channels,
        fit_mode=config.fit_mode,
        max_images=config.max_images,
    )
    _print_preprocess_summary(dataset, config.fit_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    loader_iter: Iterable[tuple[Tensor, Tensor]] = cycle(dataloader)
    effective_train_steps = resolve_train_steps(
        dataset_size=len(dataset),
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        epochs=config.epochs,
    )

    model = UnconditionalUNet(
        in_channels=dataset.channels,
        base_channels=config.base_channels,
        time_dim=config.time_dim,
    ).to(device)
    sample_image_shape = tuple(int(value) for value in dataset[0][0].shape)
    sample_channels, sample_height, sample_width = sample_image_shape
    custom_betas = build_beta_schedule(config)
    scheduler = UnconditionalDDPMScheduler(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
        betas=custom_betas,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    config.output_dir.mkdir(parents=True, exist_ok=True)
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
    train_batch_dir: Path | None = None
    print(
        f"dataset={len(dataset)} images steps={effective_train_steps} "
        f"epochs={config.epochs} device={device} beta_schedule={config.beta_schedule} "
        f"diffusion_timesteps={scheduler.timesteps} "
        f"sample_image_shape={sample_image_shape} output={config.output_dir}"
    )

    for step in range(1, effective_train_steps + 1):
        images, indices = next(loader_iter)
        images = images.to(device=device, non_blocking=True)
        if config.fit_mode != "height-flatten":
            images = _ensure_square_batch(images)
        indices = indices.to(device=device, non_blocking=True)

        if config.save_train_batches_every > 0 and step % config.save_train_batches_every == 0:
            train_batch_dir = config.output_dir / "train_batches"
            train_batch_path = save_train_batch_grid(
                images.detach().cpu(),
                indices.detach().cpu(),
                dataset,
                train_batch_dir,
                step,
            )
            print(f"saved train batch: {train_batch_path}")

        timesteps = torch.randint(0, scheduler.timesteps, (images.shape[0],), device=device)
        noise = torch.randn_like(images)
        noised = scheduler.q_sample(images, timesteps, noise)
        pred_noise = model(noised, timesteps)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step == 1 or step % config.log_every == 0 or step == effective_train_steps:
            print(f"step={step:06d} loss={last_loss:.6f}")

        if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
            checkpoint_path = save_checkpoint(model, optimizer, step, last_loss, config)
            print(f"saved checkpoint: {checkpoint_path}")

        if config.sample_every > 0 and step % config.sample_every == 0:
            samples = scheduler.sample(
                model,
                (config.sample_count, sample_channels, sample_height, sample_width),
                device,
            )
            sample_indices = _sample_source_indices(dataset, config.sample_count, device)
            sample_path = config.output_dir / "samples" / f"step_{step:06d}.png"
            save_unconditional_images(samples, sample_indices, dataset, sample_path)
            print(f"saved samples: {sample_path}")
            model.train()

    final_checkpoint = save_checkpoint(model, optimizer, effective_train_steps, last_loss, config)
    final_samples = scheduler.sample(
        model,
        (config.sample_count, sample_channels, sample_height, sample_width),
        device,
    )
    final_indices = _sample_source_indices(dataset, config.sample_count, device)
    final_sample_path = config.output_dir / "samples" / "final.png"
    save_unconditional_images(final_samples, final_indices, dataset, final_sample_path)

    process_trace_paths: dict[str, list[Path]] | None = None
    if config.save_process_traces:
        process_trace_paths = save_unconditional_process_traces(
            model,
            scheduler,
            dataset,
            config,
            device,
        )
        print(f"saved process traces: {config.output_dir / 'process_traces'}")

    return {
        "dataset_size": len(dataset),
        "train_steps": effective_train_steps,
        "final_loss": last_loss,
        "checkpoint": final_checkpoint,
        "sample_grid": final_sample_path,
        "beta_schedule": beta_schedule_path,
        "train_batches": train_batch_dir,
        "process_traces": (
            None if process_trace_paths is None else config.output_dir / "process_traces"
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an unconditional DDPM.")
    parser.add_argument("--data-root", type=Path, default=UnconditionalDDPMTrainConfig.data_root)
    parser.add_argument("--json-root", type=Path, default=UnconditionalDDPMTrainConfig.json_root)
    parser.add_argument("--output-dir", type=Path, default=UnconditionalDDPMTrainConfig.output_dir)
    parser.add_argument("--image-size", type=int, default=UnconditionalDDPMTrainConfig.image_size)
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3, 4),
        default=UnconditionalDDPMTrainConfig.channels,
    )
    parser.add_argument(
        "--fit-mode",
        choices=("pad", "resize", "reshape", "height-flatten"),
        default=UnconditionalDDPMTrainConfig.fit_mode,
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=UnconditionalDDPMTrainConfig.batch_size)
    parser.add_argument(
        "--train-steps",
        type=int,
        default=UnconditionalDDPMTrainConfig.train_steps,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--timesteps",
        type=_parse_timesteps_arg,
        default=UnconditionalDDPMTrainConfig.timesteps,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=UnconditionalDDPMTrainConfig.learning_rate,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=UnconditionalDDPMTrainConfig.base_channels,
    )
    parser.add_argument("--time-dim", type=int, default=UnconditionalDDPMTrainConfig.time_dim)
    parser.add_argument("--beta-start", type=float, default=UnconditionalDDPMTrainConfig.beta_start)
    parser.add_argument("--beta-end", type=float, default=UnconditionalDDPMTrainConfig.beta_end)
    parser.add_argument(
        "--beta-schedule",
        choices=("linear", "file", "hash-approach1", "hash-approach2"),
        default=UnconditionalDDPMTrainConfig.beta_schedule,
    )
    parser.add_argument("--beta-values-path", type=Path, default=None)
    parser.add_argument(
        "--beta-schedule-step",
        default=UnconditionalDDPMTrainConfig.beta_schedule_step,
    )
    parser.add_argument("--device", default=UnconditionalDDPMTrainConfig.device)
    parser.add_argument("--seed", type=int, default=UnconditionalDDPMTrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=UnconditionalDDPMTrainConfig.num_workers)
    parser.add_argument("--log-every", type=int, default=UnconditionalDDPMTrainConfig.log_every)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=UnconditionalDDPMTrainConfig.sample_every,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=UnconditionalDDPMTrainConfig.checkpoint_every,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=UnconditionalDDPMTrainConfig.sample_count,
    )
    parser.add_argument(
        "--save-train-batches-every",
        type=int,
        default=UnconditionalDDPMTrainConfig.save_train_batches_every,
    )
    parser.add_argument(
        "--save-process-traces",
        action="store_true",
        default=UnconditionalDDPMTrainConfig.save_process_traces,
    )
    parser.add_argument(
        "--trace-sample-count",
        type=int,
        default=UnconditionalDDPMTrainConfig.trace_sample_count,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> UnconditionalDDPMTrainConfig:
    return UnconditionalDDPMTrainConfig(
        data_root=args.data_root,
        json_root=args.json_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        channels=args.channels,
        fit_mode=args.fit_mode,
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
        save_train_batches_every=args.save_train_batches_every,
        save_process_traces=args.save_process_traces,
        trace_sample_count=args.trace_sample_count,
    )


def main() -> None:
    result = train_unconditional_ddpm(config_from_args(build_arg_parser().parse_args()))
    print(
        "done "
        f"dataset={result['dataset_size']} "
        f"checkpoint={result['checkpoint']} "
        f"samples={result['sample_grid']}"
    )
    if result["process_traces"] is not None:
        print(f"process_traces: {result['process_traces']}")


if __name__ == "__main__":
    main()
