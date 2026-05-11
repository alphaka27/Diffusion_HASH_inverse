"""
Guided conditional DDPM training pipelines.

This module keeps ``conditional_diffusion.py`` unchanged and reuses its dataset,
U-Net, beta scheduler, checkpoint, and image saving utilities.

Two guidance variants are implemented:

* classifier guidance: trains an unconditional denoiser plus a noisy-image
  classifier, then uses ``grad_x log p(label | x_t)`` during reverse sampling.
* classifier-free guidance: trains one conditional denoiser with random label
  dropout to an extra null label, then mixes conditional and unconditional noise
  predictions during reverse sampling.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_hash_inv.models.conditional_diffusion import (
    ConditionalDiffusionTrainConfig,
    ConditionalUNet,
    DDPMNoiseScheduler,
    GeneratedImageDataset,
    IndexedGeneratedImageDataset,
    SinusoidalTimeEmbedding,
    _condition_tensors_for_sampling,
    _ensure_square_batch,
    _group_count,
    _parse_timesteps_arg,
    _print_preprocess_summary,
    _reference_images_for_labels,
    _save_forward_process_trace,
    build_beta_schedule,
    resolve_device,
    resolve_train_steps,
    save_beta_schedule,
    save_image_grid,
    save_source_generated_grid,
    save_train_batch_grid,
    set_seed,
)

GuidanceMode = Literal["classifier", "classifier-free"]


@dataclass(frozen=True)
class GuidedConditionalDiffusionTrainConfig(ConditionalDiffusionTrainConfig):
    """Runtime configuration for guided conditional diffusion training."""

    output_dir: Path = Path("output/guided_conditional_diffusion")
    guidance_mode: GuidanceMode = "classifier-free"
    guidance_scale: float = 2.0
    condition_dropout: float = 0.1
    classifier_base_channels: int = 32
    classifier_learning_rate: float = 2e-4


class NoisyImageClassifier(nn.Module):
    """Classifier trained on noisy DDPM states ``x_t`` and timesteps."""

    def __init__(
        self,
        in_channels: int,
        num_conditions: int,
        base_channels: int = 32,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_conditions <= 0:
            raise ValueError("num_conditions must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.in_channels = in_channels
        self.num_conditions = num_conditions
        self.base_channels = base_channels
        self.time_dim = time_dim

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.time_projection = nn.Linear(time_dim, base_channels)
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
        self.down1 = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
        )
        self.down2 = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels * 2), base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_group_count(base_channels * 4), base_channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, num_conditions),
        )

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        emb = self.time_embedding(timesteps)
        h = self.input(x)
        h = h + self.time_projection(emb)[:, :, None, None]
        h = self.block1(h)
        h = self.down1(h)
        h = self.down2(h)
        return self.head(h)


def apply_condition_dropout(labels: Tensor, null_label: int, dropout: float) -> Tensor:
    """Replace labels with ``null_label`` with probability ``dropout``."""

    if dropout < 0.0 or dropout > 1.0:
        raise ValueError("condition_dropout must be in the interval [0, 1]")
    if dropout == 0.0:
        return labels
    null_labels = torch.full_like(labels, int(null_label))
    mask = torch.rand(labels.shape, device=labels.device) < dropout
    return torch.where(mask, null_labels, labels)


def _posterior_mean(
    scheduler: DDPMNoiseScheduler,
    x: Tensor,
    timesteps: Tensor,
    pred_noise: Tensor,
) -> Tensor:
    beta_t = scheduler._extract(scheduler.betas, timesteps, x.ndim)
    alpha_t = scheduler._extract(scheduler.alphas, timesteps, x.ndim)
    alpha_bar_t = scheduler._extract(scheduler.alpha_bars, timesteps, x.ndim)
    return (1.0 / torch.sqrt(alpha_t)) * (
        x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
    )


def _posterior_variance(
    scheduler: DDPMNoiseScheduler,
    x: Tensor,
    timesteps: Tensor,
) -> Tensor:
    return scheduler._extract(scheduler.posterior_variance, timesteps, x.ndim)


@torch.no_grad()
def p_sample_classifier_free_guidance(
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    x: Tensor,
    step: int,
    labels: Tensor,
    *,
    loop_meta: Tensor | None = None,
    null_label: int,
    guidance_scale: float,
) -> Tensor:
    """One reverse DDPM step using classifier-free guidance."""

    timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
    null_labels = torch.full_like(labels, int(null_label))
    eps_uncond = model(x, timesteps, null_labels, loop_meta)
    eps_cond = model(x, timesteps, labels, loop_meta)
    pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    mean = _posterior_mean(scheduler, x, timesteps, pred_noise)
    if step == 0:
        return mean
    variance = _posterior_variance(scheduler, x, timesteps)
    return mean + torch.sqrt(variance) * torch.randn_like(x)


def _classifier_log_prob_gradient(
    classifier: NoisyImageClassifier,
    x: Tensor,
    timesteps: Tensor,
    labels: Tensor,
) -> Tensor:
    x_in = x.detach().requires_grad_(True)
    with torch.enable_grad():
        logits = classifier(x_in, timesteps)
        selected = F.log_softmax(logits, dim=1).gather(1, labels[:, None]).sum()
        grad = torch.autograd.grad(selected, x_in)[0]
    return grad.detach()


def p_sample_classifier_guidance(
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    classifier: NoisyImageClassifier,
    x: Tensor,
    step: int,
    labels: Tensor,
    *,
    loop_meta: Tensor | None = None,
    guidance_scale: float,
) -> Tensor:
    """One reverse DDPM step using classifier guidance."""

    timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
    model_labels = torch.zeros_like(labels)
    with torch.no_grad():
        pred_noise = model(x, timesteps, model_labels, loop_meta)
        mean = _posterior_mean(scheduler, x, timesteps, pred_noise)
        variance = _posterior_variance(scheduler, x, timesteps)

    if guidance_scale != 0.0:
        grad = _classifier_log_prob_gradient(classifier, x, timesteps, labels)
        mean = mean + variance * guidance_scale * grad

    if step == 0:
        return mean
    return mean + torch.sqrt(variance) * torch.randn_like(x)


@torch.no_grad()
def sample_classifier_free_guidance(
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    shape: tuple[int, int, int, int],
    labels: Tensor,
    *,
    loop_meta: Tensor | None = None,
    null_label: int,
    guidance_scale: float,
) -> Tensor:
    x = torch.randn(shape, device=labels.device)
    model.eval()
    for step in reversed(range(scheduler.timesteps)):
        x = p_sample_classifier_free_guidance(
            scheduler,
            model,
            x,
            step,
            labels,
            loop_meta=loop_meta,
            null_label=null_label,
            guidance_scale=guidance_scale,
        )
    return x.clamp(-1.0, 1.0)


def sample_classifier_guidance(
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    classifier: NoisyImageClassifier,
    shape: tuple[int, int, int, int],
    labels: Tensor,
    *,
    loop_meta: Tensor | None = None,
    guidance_scale: float,
) -> Tensor:
    x = torch.randn(shape, device=labels.device)
    model.eval()
    classifier.eval()
    for step in reversed(range(scheduler.timesteps)):
        x = p_sample_classifier_guidance(
            scheduler,
            model,
            classifier,
            x,
            step,
            labels,
            loop_meta=loop_meta,
            guidance_scale=guidance_scale,
        )
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def sample_classifier_free_guidance_with_trace(
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    shape: tuple[int, int, int, int],
    labels: Tensor,
    *,
    loop_meta: Tensor | None = None,
    null_label: int,
    guidance_scale: float,
) -> tuple[Tensor, dict[int, Tensor]]:
    """Classifier-free guided reverse diffusion returning per-step traces."""
    x = torch.randn(shape, device=labels.device)
    model.eval()
    traces: dict[int, Tensor] = {scheduler.timesteps: x.clamp(-1.0, 1.0).detach().cpu()}
    for step in reversed(range(scheduler.timesteps)):
        x = p_sample_classifier_free_guidance(
            scheduler,
            model,
            x,
            step,
            labels,
            loop_meta=loop_meta,
            null_label=null_label,
            guidance_scale=guidance_scale,
        )
        traces[step] = x.clamp(-1.0, 1.0).detach().cpu()
    return x.clamp(-1.0, 1.0), traces


@torch.no_grad()
def sample_classifier_guidance_with_trace(
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    classifier: NoisyImageClassifier,
    shape: tuple[int, int, int, int],
    labels: Tensor,
    *,
    loop_meta: Tensor | None = None,
    guidance_scale: float,
) -> tuple[Tensor, dict[int, Tensor]]:
    """Classifier-guided reverse diffusion returning per-step traces."""
    x = torch.randn(shape, device=labels.device)
    model.eval()
    classifier.eval()
    traces: dict[int, Tensor] = {scheduler.timesteps: x.clamp(-1.0, 1.0).detach().cpu()}
    for step in reversed(range(scheduler.timesteps)):
        x = p_sample_classifier_guidance(
            scheduler,
            model,
            classifier,
            x,
            step,
            labels,
            loop_meta=loop_meta,
            guidance_scale=guidance_scale,
        )
        traces[step] = x.clamp(-1.0, 1.0).detach().cpu()
    return x.clamp(-1.0, 1.0), traces


def _save_guided_reverse_process_trace(
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    classifier: NoisyImageClassifier | None,
    config: GuidedConditionalDiffusionTrainConfig,
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

    if config.guidance_mode == "classifier-free":
        _, traces = sample_classifier_free_guidance_with_trace(
            scheduler,
            model,
            sample_shape,
            labels,
            loop_meta=loop_meta,
            null_label=len(condition_names),
            guidance_scale=config.guidance_scale,
        )
    else:
        if classifier is None:
            raise ValueError("classifier guidance requires a classifier")
        _, traces = sample_classifier_guidance_with_trace(
            scheduler,
            model,
            classifier,
            sample_shape,
            labels,
            loop_meta=loop_meta,
            guidance_scale=config.guidance_scale,
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


def save_guided_process_traces(
    model: ConditionalUNet,
    classifier: NoisyImageClassifier | None,
    scheduler: DDPMNoiseScheduler,
    dataset: GeneratedImageDataset,
    config: GuidedConditionalDiffusionTrainConfig,
    device: torch.device,
) -> dict[str, list[Path]]:
    """Save forward noising and guided reverse denoising traces to output_dir/process_traces/."""
    if config.trace_sample_count <= 0:
        raise ValueError("trace_sample_count must be positive")

    sample_count = min(config.trace_sample_count, len(dataset))
    images, labels_list, loop_metas = [], [], []
    for index in range(sample_count):
        image, label, loop_meta = dataset[index]
        images.append(image)
        labels_list.append(int(label))
        loop_metas.append(loop_meta)

    image_batch = torch.stack(images).to(device)
    label_batch = torch.tensor(labels_list, dtype=torch.long, device=device)
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
    reverse_paths = _save_guided_reverse_process_trace(
        scheduler,
        model,
        classifier,
        config,
        label_batch,
        dataset.condition_names,
        trace_dir / "reverse",
        sample_shape,
        loop_meta=loop_meta_batch,
    )
    return {"forward": forward_paths, "reverse": reverse_paths}


def _validate_config(config: GuidedConditionalDiffusionTrainConfig) -> None:
    if config.guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if config.condition_dropout < 0.0 or config.condition_dropout > 1.0:
        raise ValueError("condition_dropout must be in the interval [0, 1]")
    if config.classifier_base_channels <= 0:
        raise ValueError("classifier_base_channels must be positive")
    if config.classifier_learning_rate <= 0.0:
        raise ValueError("classifier_learning_rate must be positive")


def _dataset_from_config(config: GuidedConditionalDiffusionTrainConfig) -> GeneratedImageDataset:
    return GeneratedImageDataset(
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


def _jsonable_config(config: GuidedConditionalDiffusionTrainConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["data_root"] = str(config.data_root)
    payload["json_root"] = str(config.json_root)
    payload["output_dir"] = str(config.output_dir)
    payload["beta_values_path"] = (
        None if config.beta_values_path is None else str(config.beta_values_path)
    )
    return payload


def _save_guided_checkpoint(
    *,
    model: ConditionalUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    diffusion_loss: float,
    classifier_loss: float | None,
    dataset: GeneratedImageDataset,
    config: GuidedConditionalDiffusionTrainConfig,
    classifier: NoisyImageClassifier | None = None,
    classifier_optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    path = config.output_dir / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "step": step,
        "guidance_mode": config.guidance_mode,
        "guidance_scale": config.guidance_scale,
        "diffusion_loss": diffusion_loss,
        "classifier_loss": classifier_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "condition_to_idx": dataset.condition_to_idx,
        "config": _jsonable_config(config),
        "model_args": {
            "in_channels": model.in_channels,
            "num_conditions": model.num_conditions,
            "base_channels": model.base_channels,
            "time_dim": model.time_dim,
            "temporal_conditioning": model.temporal_conditioning,
            "max_loop_count": model.max_loop_count,
        },
    }
    if config.guidance_mode == "classifier-free":
        payload["null_label"] = dataset.num_conditions
    if classifier is not None:
        payload["classifier_state_dict"] = classifier.state_dict()
        payload["classifier_args"] = {
            "in_channels": classifier.in_channels,
            "num_conditions": classifier.num_conditions,
            "base_channels": classifier.base_channels,
            "time_dim": classifier.time_dim,
        }
    if classifier_optimizer is not None:
        payload["classifier_optimizer_state_dict"] = classifier_optimizer.state_dict()
    torch.save(payload, path)
    return path


def _sample_guided(
    *,
    config: GuidedConditionalDiffusionTrainConfig,
    scheduler: DDPMNoiseScheduler,
    model: ConditionalUNet,
    classifier: NoisyImageClassifier | None,
    dataset: GeneratedImageDataset,
    sample_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    labels, loop_meta = _condition_tensors_for_sampling(dataset, config.sample_count, device)
    shape = (labels.shape[0], *sample_shape)
    if config.guidance_mode == "classifier-free":
        samples = sample_classifier_free_guidance(
            scheduler,
            model,
            shape,
            labels,
            loop_meta=loop_meta,
            null_label=dataset.num_conditions,
            guidance_scale=config.guidance_scale,
        )
        return samples, labels

    if classifier is None:
        raise ValueError("classifier guidance requires a classifier")
    samples = sample_classifier_guidance(
        scheduler,
        model,
        classifier,
        shape,
        labels,
        loop_meta=loop_meta,
        guidance_scale=config.guidance_scale,
    )
    return samples, labels


def train_guided_conditional_diffusion(
    config: GuidedConditionalDiffusionTrainConfig,
) -> dict[str, Path | float | int | str | None]:
    """Train and sample a guided conditional DDPM."""

    _validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = _dataset_from_config(config)
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
    loader_iter = cycle(dataloader)
    effective_train_steps = resolve_train_steps(
        dataset_size=len(dataset),
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        epochs=config.epochs,
    )

    if config.guidance_mode == "classifier-free":
        model_num_conditions = dataset.num_conditions + 1
    else:
        model_num_conditions = 1
    model = ConditionalUNet(
        in_channels=dataset.channels,
        num_conditions=model_num_conditions,
        base_channels=config.base_channels,
        time_dim=config.time_dim,
        temporal_conditioning=config.temporal_conditioning,
        max_loop_count=config.max_loop_count,
    ).to(device)
    classifier: NoisyImageClassifier | None = None
    classifier_optimizer: torch.optim.Optimizer | None = None
    if config.guidance_mode == "classifier":
        classifier = NoisyImageClassifier(
            in_channels=dataset.channels,
            num_conditions=dataset.num_conditions,
            base_channels=config.classifier_base_channels,
            time_dim=config.time_dim,
        ).to(device)
        classifier_optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=config.classifier_learning_rate,
        )

    sample_image_shape = tuple(int(value) for value in dataset[0][0].shape)
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
    (config.output_dir / "train_config.json").write_text(
        json.dumps(_jsonable_config(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    beta_schedule_path = save_beta_schedule(scheduler, config)

    last_diffusion_loss = math.nan
    last_classifier_loss: float | None = None
    print(
        f"dataset={len(dataset)} images conditions={dataset.num_conditions} "
        f"guidance_mode={config.guidance_mode} guidance_scale={config.guidance_scale} "
        f"label_source={config.label_source} steps={effective_train_steps} "
        f"epochs={config.epochs} device={device} beta_schedule={config.beta_schedule} "
        f"diffusion_timesteps={scheduler.timesteps} sample_image_shape={sample_image_shape} "
        f"output={config.output_dir}"
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

        if config.guidance_mode == "classifier-free":
            train_labels = apply_condition_dropout(
                labels,
                null_label=dataset.num_conditions,
                dropout=config.condition_dropout,
            )
            pred_noise = model(noised, timesteps, train_labels, loop_meta)
        else:
            pred_noise = model(noised, timesteps, torch.zeros_like(labels), loop_meta)
        diffusion_loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        diffusion_loss.backward()
        optimizer.step()
        last_diffusion_loss = float(diffusion_loss.detach().cpu())

        if classifier is not None and classifier_optimizer is not None:
            logits = classifier(noised.detach(), timesteps)
            classifier_loss = F.cross_entropy(logits, labels)
            classifier_optimizer.zero_grad(set_to_none=True)
            classifier_loss.backward()
            classifier_optimizer.step()
            last_classifier_loss = float(classifier_loss.detach().cpu())

        if step == 1 or step % config.log_every == 0 or step == effective_train_steps:
            message = f"step={step:06d} diffusion_loss={last_diffusion_loss:.6f}"
            if last_classifier_loss is not None:
                message += f" classifier_loss={last_classifier_loss:.6f}"
            print(message)

        if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
            checkpoint_path = _save_guided_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                diffusion_loss=last_diffusion_loss,
                classifier_loss=last_classifier_loss,
                dataset=dataset,
                config=config,
                classifier=classifier,
                classifier_optimizer=classifier_optimizer,
            )
            print(f"saved checkpoint: {checkpoint_path}")

        if config.sample_every > 0 and step % config.sample_every == 0:
            samples, sample_labels = _sample_guided(
                config=config,
                scheduler=scheduler,
                model=model,
                classifier=classifier,
                dataset=dataset,
                sample_shape=sample_image_shape,
                device=device,
            )
            sample_path = config.output_dir / "samples" / f"step_{step:06d}.png"
            save_image_grid(samples, sample_labels, dataset.condition_names, sample_path)
            reference_images = _reference_images_for_labels(dataset, sample_labels)
            paired_sample_path = config.output_dir / "samples" / f"step_{step:06d}.with_source.png"
            save_source_generated_grid(
                reference_images,
                samples,
                sample_labels,
                dataset.condition_names,
                paired_sample_path,
            )
            print(f"saved samples: {sample_path}")
            print(f"saved source+generated samples: {paired_sample_path}")
            model.train()
            if classifier is not None:
                classifier.train()

    final_checkpoint = _save_guided_checkpoint(
        model=model,
        optimizer=optimizer,
        step=effective_train_steps,
        diffusion_loss=last_diffusion_loss,
        classifier_loss=last_classifier_loss,
        dataset=dataset,
        config=config,
        classifier=classifier,
        classifier_optimizer=classifier_optimizer,
    )
    samples, sample_labels = _sample_guided(
        config=config,
        scheduler=scheduler,
        model=model,
        classifier=classifier,
        dataset=dataset,
        sample_shape=sample_image_shape,
        device=device,
    )
    final_sample_path = config.output_dir / "samples" / "final.png"
    save_image_grid(samples, sample_labels, dataset.condition_names, final_sample_path)
    final_source_path = config.output_dir / "samples" / "final.source.png"
    reference_images = _reference_images_for_labels(dataset, sample_labels)
    save_image_grid(reference_images, sample_labels, dataset.condition_names, final_source_path)
    final_with_source_path = config.output_dir / "samples" / "final.with_source.png"
    save_source_generated_grid(
        reference_images,
        samples,
        sample_labels,
        dataset.condition_names,
        final_with_source_path,
    )

    process_trace_paths: dict[str, list[Path]] | None = None
    if config.save_process_traces:
        process_trace_paths = save_guided_process_traces(
            model,
            classifier,
            scheduler,
            dataset,
            config,
            device,
        )
        print(f"saved process traces: {config.output_dir / 'process_traces'}")

    return {
        "dataset_size": len(dataset),
        "num_conditions": dataset.num_conditions,
        "guidance_mode": config.guidance_mode,
        "guidance_scale": config.guidance_scale,
        "train_steps": effective_train_steps,
        "final_loss": last_diffusion_loss,
        "final_classifier_loss": last_classifier_loss,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a classifier-guided or classifier-free guided conditional DDPM."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=GuidedConditionalDiffusionTrainConfig.data_root,
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=GuidedConditionalDiffusionTrainConfig.json_root,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GuidedConditionalDiffusionTrainConfig.output_dir,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.image_size,
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3, 4),
        default=GuidedConditionalDiffusionTrainConfig.channels,
    )
    parser.add_argument(
        "--fit-mode",
        choices=("pad", "resize", "reshape", "height-flatten"),
        default=GuidedConditionalDiffusionTrainConfig.fit_mode,
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
        default=GuidedConditionalDiffusionTrainConfig.condition_mode,
    )
    parser.add_argument(
        "--label-source",
        choices=("final-hash",),
        default=GuidedConditionalDiffusionTrainConfig.label_source,
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.batch_size,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.train_steps,
    )
    parser.add_argument("--epochs", type=int, default=GuidedConditionalDiffusionTrainConfig.epochs)
    parser.add_argument(
        "--timesteps",
        type=_parse_timesteps_arg,
        default=GuidedConditionalDiffusionTrainConfig.timesteps,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=GuidedConditionalDiffusionTrainConfig.learning_rate,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.base_channels,
    )
    parser.add_argument(
        "--time-dim",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.time_dim,
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=GuidedConditionalDiffusionTrainConfig.beta_start,
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=GuidedConditionalDiffusionTrainConfig.beta_end,
    )
    parser.add_argument(
        "--beta-schedule",
        choices=("linear", "file", "hash-approach1", "hash-approach2"),
        default=GuidedConditionalDiffusionTrainConfig.beta_schedule,
    )
    parser.add_argument(
        "--beta-values-path",
        type=Path,
        default=GuidedConditionalDiffusionTrainConfig.beta_values_path,
    )
    parser.add_argument(
        "--beta-schedule-step",
        default=GuidedConditionalDiffusionTrainConfig.beta_schedule_step,
    )
    parser.add_argument("--device", default=GuidedConditionalDiffusionTrainConfig.device)
    parser.add_argument("--seed", type=int, default=GuidedConditionalDiffusionTrainConfig.seed)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.num_workers,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.log_every,
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.sample_every,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.checkpoint_every,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.sample_count,
    )
    parser.add_argument(
        "--save-train-batches-every",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.save_train_batches_every,
    )
    parser.add_argument(
        "--save-process-traces",
        action="store_true",
        default=GuidedConditionalDiffusionTrainConfig.save_process_traces,
    )
    parser.add_argument(
        "--trace-sample-count",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.trace_sample_count,
    )
    parser.add_argument(
        "--temporal-conditioning",
        choices=("class", "loop-sinusoidal", "loop-structured", "loop-sequence"),
        default=GuidedConditionalDiffusionTrainConfig.temporal_conditioning,
    )
    parser.add_argument(
        "--use-loop-images",
        action="store_true",
        default=GuidedConditionalDiffusionTrainConfig.use_loop_images,
    )
    parser.add_argument(
        "--max-loop-count",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.max_loop_count,
    )
    parser.add_argument(
        "--guidance-mode",
        choices=("classifier", "classifier-free"),
        default=GuidedConditionalDiffusionTrainConfig.guidance_mode,
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=GuidedConditionalDiffusionTrainConfig.guidance_scale,
    )
    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=GuidedConditionalDiffusionTrainConfig.condition_dropout,
        help="Only used for classifier-free guidance training.",
    )
    parser.add_argument(
        "--classifier-base-channels",
        type=int,
        default=GuidedConditionalDiffusionTrainConfig.classifier_base_channels,
        help="Only used for classifier guidance.",
    )
    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=GuidedConditionalDiffusionTrainConfig.classifier_learning_rate,
        help="Only used for classifier guidance.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> GuidedConditionalDiffusionTrainConfig:
    return GuidedConditionalDiffusionTrainConfig(
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
        save_train_batches_every=args.save_train_batches_every,
        save_process_traces=args.save_process_traces,
        trace_sample_count=args.trace_sample_count,
        temporal_conditioning=args.temporal_conditioning,
        use_loop_images=args.use_loop_images,
        max_loop_count=args.max_loop_count,
        guidance_mode=args.guidance_mode,
        guidance_scale=args.guidance_scale,
        condition_dropout=args.condition_dropout,
        classifier_base_channels=args.classifier_base_channels,
        classifier_learning_rate=args.classifier_learning_rate,
    )


def main() -> None:
    result = train_guided_conditional_diffusion(config_from_args(build_arg_parser().parse_args()))
    print(
        "completed "
        f"guidance_mode={result['guidance_mode']} "
        f"dataset_size={result['dataset_size']} "
        f"conditions={result['num_conditions']} "
        f"train_steps={result['train_steps']} "
        f"final_loss={result['final_loss']:.6f}"
    )
    if result["final_classifier_loss"] is not None:
        print(f"final_classifier_loss: {result['final_classifier_loss']:.6f}")
    print(f"checkpoint: {result['checkpoint']}")
    print(f"sample_grid: {result['sample_grid']}")
    print(f"beta_schedule: {result['beta_schedule']}")
    if result["train_batches"] is not None:
        print(f"train_batches: {result['train_batches']}")
    if result["process_traces"] is not None:
        print(f"process_traces: {result['process_traces']}")


if __name__ == "__main__":
    main()
