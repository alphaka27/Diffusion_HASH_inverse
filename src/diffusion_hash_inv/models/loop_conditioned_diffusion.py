"""
DDPM training pipeline with structured MD5 loop-state conditioning.

Unlike ``conditional_diffusion.py``, this module does not collapse
``Logs/<step>`` into one categorical class id. It reads the 64 loop states from
one hash step, converts the A/B/C/D 32-bit words into a continuous tensor, and
feeds that tensor through a condition encoder inside the denoising U-Net.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_hash_inv.models.conditional_diffusion import (
    ConditionalDiffusionTrainConfig,
    ConditionalResBlock,
    DDPMNoiseScheduler,
    FitMode,
    SinusoidalTimeEmbedding,
    _denormalize_images,
    _ensure_square_batch,
    _group_count,
    _image_from_tensor,
    _load_json_index,
    _lookup_nested_value,
    _parse_timesteps_arg,
    _print_preprocess_summary,
    _read_json_payload,
    _save_forward_process_trace,
    build_beta_schedule,
    resolve_device,
    resolve_train_steps,
    save_beta_schedule,
    save_image_grid,
    set_seed,
)
from diffusion_hash_inv.models.conditional_diffusion import (
    _fit_image,
    _normalize_image_array,
)

WORD_MAX = float(0xFFFFFFFF)
DEFAULT_WORD_NAMES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class LoopConditionedSample:
    """One generated image paired with a structured loop condition tensor."""

    path: Path
    run_id: str
    condition: np.ndarray


@dataclass(frozen=True)
class LoopConditionedDiffusionTrainConfig(ConditionalDiffusionTrainConfig):
    """Runtime configuration for structured loop-conditioned DDPM training."""

    output_dir: Path = Path("output/loop_conditioned_diffusion")
    condition_step: str = "4th Step"
    condition_round: str = "1st Round"
    loop_count: int = 64
    word_names: tuple[str, ...] = DEFAULT_WORD_NAMES


def _ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 20:
        return "th"
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return suffixes.get(value % 10, "th")


def loop_key(index: int) -> str:
    """Return the JSON key used by ``StepLogs.index_label(index, "Loop")``."""

    if index <= 0:
        raise ValueError("loop index must be positive")
    return f"{index}{_ordinal_suffix(index)} Loop"


def _parse_hex_word(value: object) -> int:
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = int(text, 16 if text.lower().startswith("0x") else 10)
        except ValueError as exc:
            raise ValueError(f"Invalid loop word value: {value!r}") from exc
    else:
        raise TypeError(f"Loop word value must be int or string, got {type(value).__name__}")

    if parsed < 0 or parsed > 0xFFFFFFFF:
        raise ValueError(f"Loop word value out of uint32 range: {parsed}")
    return parsed


def _normalize_uint32(value: int) -> float:
    return (float(value) / WORD_MAX) * 2.0 - 1.0


def extract_loop_condition(
    payload: dict[str, object],
    *,
    condition_step: str = "4th Step",
    condition_round: str = "1st Round",
    loop_count: int = 64,
    word_names: Sequence[str] = DEFAULT_WORD_NAMES,
) -> np.ndarray:
    """
    Extract ``loop_count x len(word_names)`` normalized loop-state features.

    The returned array is ``float32`` in ``[-1, 1]``. For the default MD5 trace
    format this means 64 loops by A/B/C/D words, giving a ``(64, 4)`` condition
    tensor per training sample.
    """

    if loop_count <= 0:
        raise ValueError("loop_count must be positive")
    if not word_names:
        raise ValueError("word_names must not be empty")

    round_payload = _lookup_nested_value(payload, ("Logs", condition_step, condition_round))
    if not isinstance(round_payload, dict):
        raise ValueError(f"Logs/{condition_step}/{condition_round} must be a JSON object")

    rows: list[list[float]] = []
    for loop_idx in range(1, loop_count + 1):
        key = loop_key(loop_idx)
        loop_payload = round_payload.get(key)
        if not isinstance(loop_payload, dict):
            raise KeyError(
                "JSON loop condition path not found: "
                f"Logs/{condition_step}/{condition_round}/{key}"
            )
        row: list[float] = []
        for word_name in word_names:
            if word_name not in loop_payload:
                raise KeyError(
                    "JSON loop condition word not found: "
                    f"Logs/{condition_step}/{condition_round}/{key}/{word_name}"
                )
            row.append(_normalize_uint32(_parse_hex_word(loop_payload[word_name])))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def discover_loop_conditioned_samples(
    root: Path | str,
    *,
    json_root: Path | str = Path("output/json"),
    condition_step: str = "4th Step",
    condition_round: str = "1st Round",
    loop_count: int = 64,
    word_names: Sequence[str] = DEFAULT_WORD_NAMES,
    max_images: int | None = None,
) -> list[LoopConditionedSample]:
    """Discover ``message.png`` files and attach structured loop conditions."""

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
    samples: list[LoopConditionedSample] = []
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
        condition = extract_loop_condition(
            payload_cache[run_id],
            condition_step=condition_step,
            condition_round=condition_round,
            loop_count=loop_count,
            word_names=word_names,
        )
        samples.append(LoopConditionedSample(path=path, run_id=run_id, condition=condition))
    return samples


class LoopConditionedImageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """
    Dataset that returns ``(image, loop_condition, sample_index)``.

    ``loop_condition`` is a float tensor with shape
    ``(loop_count, len(word_names))``.
    """

    def __init__(
        self,
        root: Path | str,
        json_root: Path | str = Path("output/json"),
        image_size: int = 64,
        channels: int = 3,
        fit_mode: FitMode = "reshape",
        condition_step: str = "4th Step",
        condition_round: str = "1st Round",
        loop_count: int = 64,
        word_names: Sequence[str] = DEFAULT_WORD_NAMES,
        max_images: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.json_root = Path(json_root)
        self.image_size = image_size
        self.fit_mode = fit_mode
        self.source_channels = channels
        self.channels = channels
        self.condition_step = condition_step
        self.condition_round = condition_round
        self.loop_count = loop_count
        self.word_names = tuple(word_names)
        self.samples = discover_loop_conditioned_samples(
            self.root,
            json_root=json_root,
            condition_step=condition_step,
            condition_round=condition_round,
            loop_count=loop_count,
            word_names=self.word_names,
            max_images=max_images,
        )

    @property
    def condition_shape(self) -> tuple[int, int]:
        return self.loop_count, len(self.word_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            fitted = _fit_image(image, self.image_size, self.fit_mode, self.source_channels)
            array = _normalize_image_array(fitted, self.channels)
        image_tensor = torch.from_numpy(array).to(dtype=torch.float32)
        condition_tensor = torch.from_numpy(sample.condition).to(dtype=torch.float32)
        return image_tensor, condition_tensor, torch.tensor(index, dtype=torch.long)


class LoopConditionedUNet(nn.Module):
    """Small U-Net that predicts DDPM noise from image, timestep, and loop state."""

    def __init__(
        self,
        in_channels: int,
        condition_shape: tuple[int, int] = (64, 4),
        base_channels: int = 64,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if len(condition_shape) != 2 or condition_shape[0] <= 0 or condition_shape[1] <= 0:
            raise ValueError("condition_shape must be two positive dimensions")

        self.in_channels = in_channels
        self.condition_shape = tuple(int(value) for value in condition_shape)
        self.base_channels = base_channels
        self.time_dim = time_dim
        condition_dim = math.prod(self.condition_shape)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.condition_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(condition_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
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

    def _embedding(self, timesteps: Tensor, conditions: Tensor) -> Tensor:
        expected = (conditions.shape[0], *self.condition_shape)
        if tuple(conditions.shape) != expected:
            raise ValueError(
                f"conditions must have shape {expected}, got {tuple(conditions.shape)}"
            )
        return self.time_embedding(timesteps) + self.condition_embedding(conditions)

    @staticmethod
    def _match_spatial(x: Tensor, target: Tensor) -> Tensor:
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, x: Tensor, timesteps: Tensor, conditions: Tensor) -> Tensor:
        emb = self._embedding(timesteps, conditions)
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


class LoopConditionedDDPMScheduler(DDPMNoiseScheduler):
    """DDPM scheduler with structured-condition reverse sampling."""

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: Tensor, step: int, conditions: Tensor) -> Tensor:
        timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        beta_t = self._extract(self.betas, timesteps, x.ndim)
        alpha_t = self._extract(self.alphas, timesteps, x.ndim)
        alpha_bar_t = self._extract(self.alpha_bars, timesteps, x.ndim)
        pred_noise = model(x, timesteps, conditions)

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
        conditions: Tensor,
    ) -> Tensor:
        x = torch.randn(shape, device=conditions.device)
        model.eval()
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step, conditions)
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_with_trace(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        conditions: Tensor,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """Run full reverse diffusion, returning all per-step intermediate images."""
        x = torch.randn(shape, device=conditions.device)
        model.eval()
        traces: dict[int, Tensor] = {self.timesteps: x.clamp(-1.0, 1.0).detach().cpu()}
        for step in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, step, conditions)
            traces[step] = x.clamp(-1.0, 1.0).detach().cpu()
        return x.clamp(-1.0, 1.0), traces


def _sample_conditions_for_sampling(
    dataset: LoopConditionedImageDataset,
    sample_count: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    indices = [index for _, index in zip(range(sample_count), cycle(range(len(dataset))))]
    conditions = [dataset.samples[index].condition for index in indices]
    condition_batch = torch.from_numpy(np.stack(conditions)).to(device=device, dtype=torch.float32)
    index_batch = torch.tensor(indices, device=device, dtype=torch.long)
    return condition_batch, index_batch


def save_loop_conditioned_images(
    images: Tensor,
    sample_indices: Tensor,
    dataset: LoopConditionedImageDataset,
    path: Path,
) -> Path:
    """Save generated images and sidecar JSON with source run ids."""

    images = _denormalize_images(images)
    indices = sample_indices.detach().cpu().tolist()
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

    metadata = []
    for idx, sample_index in enumerate(indices):
        sample = dataset.samples[int(sample_index)]
        metadata.append(
            {
                "index": idx,
                "file": saved_files[idx],
                "dataset_index": int(sample_index),
                "run_id": sample.run_id,
                "source_path": str(sample.path),
                "condition_step": dataset.condition_step,
                "condition_round": dataset.condition_round,
            }
        )
    path.with_suffix(".conditions.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return path


def _checkpoint_payload(
    model: LoopConditionedUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    dataset: LoopConditionedImageDataset,
    config: LoopConditionedDiffusionTrainConfig,
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
        "condition_shape": dataset.condition_shape,
        "model_args": {
            "in_channels": model.in_channels,
            "condition_shape": model.condition_shape,
            "base_channels": model.base_channels,
            "time_dim": model.time_dim,
        },
    }


def save_checkpoint(
    model: LoopConditionedUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    dataset: LoopConditionedImageDataset,
    config: LoopConditionedDiffusionTrainConfig,
) -> Path:
    path = config.output_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, step, loss, dataset, config), path)
    return path


def save_loop_process_traces(
    model: nn.Module,
    scheduler: LoopConditionedDDPMScheduler,
    dataset: LoopConditionedImageDataset,
    config: LoopConditionedDiffusionTrainConfig,
    device: torch.device,
) -> dict[str, list[Path]]:
    """Save forward noising and reverse denoising traces to output_dir/process_traces/."""
    if config.trace_sample_count <= 0:
        raise ValueError("trace_sample_count must be positive")

    sample_count = min(config.trace_sample_count, len(dataset))
    images, conditions, indices = [], [], []
    for index in range(sample_count):
        image, condition, _idx = dataset[index]
        images.append(image)
        conditions.append(condition)
        indices.append(index)

    image_batch = torch.stack(images).to(device)
    condition_batch = torch.stack(conditions).to(device)
    label_batch = torch.tensor(indices, dtype=torch.long, device=device)
    condition_names = [f"loop_{i}" for i in range(sample_count)]
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
    _, traces = scheduler.sample_with_trace(model, sample_shape, condition_batch)
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


def train_loop_conditioned_diffusion(
    config: LoopConditionedDiffusionTrainConfig,
) -> dict[str, Path | float | int | None]:
    """Train a DDPM conditioned on structured 64-loop hash state tensors."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = LoopConditionedImageDataset(
        root=config.data_root,
        json_root=config.json_root,
        image_size=config.image_size,
        channels=config.channels,
        fit_mode=config.fit_mode,
        condition_step=config.condition_step,
        condition_round=config.condition_round,
        loop_count=config.loop_count,
        word_names=config.word_names,
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
    loader_iter: Iterable[tuple[Tensor, Tensor, Tensor]] = cycle(dataloader)
    effective_train_steps = resolve_train_steps(
        dataset_size=len(dataset),
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        epochs=config.epochs,
    )

    model = LoopConditionedUNet(
        in_channels=dataset.channels,
        condition_shape=dataset.condition_shape,
        base_channels=config.base_channels,
        time_dim=config.time_dim,
    ).to(device)
    sample_image_shape = tuple(int(value) for value in dataset[0][0].shape)
    sample_channels, sample_height, sample_width = sample_image_shape
    custom_betas = build_beta_schedule(config)
    scheduler = LoopConditionedDDPMScheduler(
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
    condition_metadata = {
        "condition_step": config.condition_step,
        "condition_round": config.condition_round,
        "loop_count": config.loop_count,
        "word_names": list(config.word_names),
        "condition_shape": list(dataset.condition_shape),
        "normalization": "uint32 / 0xffffffff * 2 - 1",
    }
    (config.output_dir / "condition_schema.json").write_text(
        json.dumps(condition_metadata, indent=2),
        encoding="utf-8",
    )
    beta_schedule_path = save_beta_schedule(scheduler, config)

    last_loss = math.nan
    print(
        f"dataset={len(dataset)} images condition_shape={dataset.condition_shape} "
        f"steps={effective_train_steps} epochs={config.epochs} "
        f"device={device} beta_schedule={config.beta_schedule} "
        f"diffusion_timesteps={scheduler.timesteps} "
        f"sample_image_shape={sample_image_shape} output={config.output_dir}"
    )

    for step in range(1, effective_train_steps + 1):
        images, conditions, _indices = next(loader_iter)
        images = images.to(device=device, non_blocking=True)
        if config.fit_mode != "height-flatten":
            images = _ensure_square_batch(images)
        conditions = conditions.to(device=device, non_blocking=True)
        timesteps = torch.randint(0, scheduler.timesteps, (images.shape[0],), device=device)
        noise = torch.randn_like(images)
        noised = scheduler.q_sample(images, timesteps, noise)
        pred_noise = model(noised, timesteps, conditions)
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
            conditions_for_sample, source_indices = _sample_conditions_for_sampling(
                dataset,
                config.sample_count,
                device,
            )
            samples = scheduler.sample(
                model,
                (
                    conditions_for_sample.shape[0],
                    sample_channels,
                    sample_height,
                    sample_width,
                ),
                conditions_for_sample,
            )
            sample_path = config.output_dir / "samples" / f"step_{step:06d}.png"
            save_loop_conditioned_images(
                samples,
                source_indices,
                dataset,
                sample_path,
            )
            print(f"saved samples: {sample_path}")
            model.train()

    final_checkpoint = save_checkpoint(
        model,
        optimizer,
        effective_train_steps,
        last_loss,
        dataset,
        config,
    )
    conditions_for_sample, source_indices = _sample_conditions_for_sampling(
        dataset,
        config.sample_count,
        device,
    )
    samples = scheduler.sample(
        model,
        (
            conditions_for_sample.shape[0],
            sample_channels,
            sample_height,
            sample_width,
        ),
        conditions_for_sample,
    )
    final_sample_path = config.output_dir / "samples" / "final.png"
    save_loop_conditioned_images(samples, source_indices, dataset, final_sample_path)

    process_trace_paths: dict[str, list[Path]] | None = None
    if config.save_process_traces:
        process_trace_paths = save_loop_process_traces(
            model,
            scheduler,
            dataset,
            config,
            device,
        )
        print(f"saved process traces: {config.output_dir / 'process_traces'}")

    return {
        "dataset_size": len(dataset),
        "condition_rows": dataset.condition_shape[0],
        "condition_columns": dataset.condition_shape[1],
        "train_steps": effective_train_steps,
        "final_loss": last_loss,
        "checkpoint": final_checkpoint,
        "sample_grid": final_sample_path,
        "beta_schedule": beta_schedule_path,
        "process_traces": (
            None if process_trace_paths is None else config.output_dir / "process_traces"
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DDPM with structured MD5 loop-state conditioning."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=LoopConditionedDiffusionTrainConfig.data_root,
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=LoopConditionedDiffusionTrainConfig.json_root,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LoopConditionedDiffusionTrainConfig.output_dir,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.image_size,
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3, 4),
        default=LoopConditionedDiffusionTrainConfig.channels,
    )
    parser.add_argument(
        "--fit-mode",
        choices=("pad", "resize", "reshape", "height-flatten"),
        default=LoopConditionedDiffusionTrainConfig.fit_mode,
    )
    parser.add_argument(
        "--condition-step",
        default=LoopConditionedDiffusionTrainConfig.condition_step,
    )
    parser.add_argument(
        "--condition-round",
        default=LoopConditionedDiffusionTrainConfig.condition_round,
    )
    parser.add_argument(
        "--loop-count",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.loop_count,
    )
    parser.add_argument(
        "--word-names",
        nargs="+",
        default=list(LoopConditionedDiffusionTrainConfig.word_names),
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.batch_size,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.train_steps,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--timesteps",
        type=_parse_timesteps_arg,
        default=LoopConditionedDiffusionTrainConfig.timesteps,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LoopConditionedDiffusionTrainConfig.learning_rate,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.base_channels,
    )
    parser.add_argument(
        "--time-dim",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.time_dim,
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=LoopConditionedDiffusionTrainConfig.beta_start,
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=LoopConditionedDiffusionTrainConfig.beta_end,
    )
    parser.add_argument(
        "--beta-schedule",
        choices=("linear", "file", "hash-approach1", "hash-approach2"),
        default=LoopConditionedDiffusionTrainConfig.beta_schedule,
    )
    parser.add_argument("--beta-values-path", type=Path, default=None)
    parser.add_argument(
        "--beta-schedule-step",
        default=LoopConditionedDiffusionTrainConfig.beta_schedule_step,
    )
    parser.add_argument("--device", default=LoopConditionedDiffusionTrainConfig.device)
    parser.add_argument("--seed", type=int, default=LoopConditionedDiffusionTrainConfig.seed)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.num_workers,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.log_every,
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.sample_every,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.checkpoint_every,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.sample_count,
    )
    parser.add_argument(
        "--save-process-traces",
        action="store_true",
        default=LoopConditionedDiffusionTrainConfig.save_process_traces,
    )
    parser.add_argument(
        "--trace-sample-count",
        type=int,
        default=LoopConditionedDiffusionTrainConfig.trace_sample_count,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> LoopConditionedDiffusionTrainConfig:
    return LoopConditionedDiffusionTrainConfig(
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
        save_process_traces=args.save_process_traces,
        trace_sample_count=args.trace_sample_count,
        condition_step=args.condition_step,
        condition_round=args.condition_round,
        loop_count=args.loop_count,
        word_names=tuple(args.word_names),
    )


def main() -> None:
    result = train_loop_conditioned_diffusion(config_from_args(build_arg_parser().parse_args()))
    print(
        "done "
        f"dataset={result['dataset_size']} "
        f"condition_shape=({result['condition_rows']}, {result['condition_columns']}) "
        f"checkpoint={result['checkpoint']} "
        f"samples={result['sample_grid']}"
    )
    if result["process_traces"] is not None:
        print(f"process_traces: {result['process_traces']}")


if __name__ == "__main__":
    main()
