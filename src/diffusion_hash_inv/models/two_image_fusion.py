"""
Minimal conditional diffusion model for fusing two images into one target image.

This module expects image tensors normalized to the range ``[-1, 1]`` and uses
two source images as conditions while denoising the target image.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class FusionNoisePredictor(nn.Module):
    """
    Predict diffusion noise from two source images and a noisy target image.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
    ):
        super().__init__()
        input_channels = image_channels * 3
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.high_block = ResidualBlock(base_channels, base_channels, time_dim)
        self.low_block = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.mid_block = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.fuse_block = ResidualBlock(base_channels * 3, base_channels, time_dim)
        self.output = nn.Sequential(
            _group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if source_a.shape != source_b.shape or source_a.shape != noisy_target.shape:
            raise ValueError("source_a, source_b, and noisy_target must have matching shapes")

        time_embedding = self.time_embedding(timesteps)
        x = torch.cat((source_a, source_b, noisy_target), dim=1)
        high = self.high_block(self.stem(x), time_embedding)
        low = F.avg_pool2d(high, kernel_size=2, stride=2)
        low = self.low_block(low, time_embedding)
        low = self.mid_block(low, time_embedding)
        low = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat((high, low), dim=1)
        return self.output(self.fuse_block(fused, time_embedding))


class TwoImageFusionDiffusion(nn.Module):
    """
    Small DDPM-style wrapper for image fusion tasks.

    Training data should be triples ``(source_a, source_b, target)`` where all
    tensors share the same shape ``[batch, channels, height, width]``.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.noise_predictor = FusionNoisePredictor(
            image_channels=image_channels,
            base_channels=base_channels,
            time_dim=time_dim,
        )

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat((torch.ones(1), alpha_bars[:-1]), dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-12))

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        extracted = values.gather(0, timesteps)
        return extracted.view(shape[0], 1, 1, 1)

    def q_sample(
        self,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(target)

        alpha_bar_t = self._extract(self.alpha_bars, timesteps, target.shape)
        return alpha_bar_t.sqrt() * target + (1.0 - alpha_bar_t).sqrt() * noise

    def training_loss(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = target.shape[0]
        device = target.device

        if timesteps is None:
            timesteps = torch.randint(0, self.num_steps, (batch_size,), device=device)
        if noise is None:
            noise = torch.randn_like(target)

        noisy_target = self.q_sample(target, timesteps, noise)
        predicted_noise = self.noise_predictor(source_a, source_b, noisy_target, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        source_a: torch.Tensor,
        source_b: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_a.shape != source_b.shape:
            raise ValueError("source_a and source_b must have matching shapes")

        current = noise if noise is not None else torch.randn_like(source_a)

        for step in range(self.num_steps - 1, -1, -1):
            timesteps = torch.full(
                (source_a.shape[0],),
                step,
                device=source_a.device,
                dtype=torch.long,
            )
            predicted_noise = self.noise_predictor(source_a, source_b, current, timesteps)

            alpha_t = self._extract(self.alphas, timesteps, current.shape)
            alpha_bar_t = self._extract(self.alpha_bars, timesteps, current.shape)
            beta_t = self._extract(self.betas, timesteps, current.shape)

            mean = (current - beta_t * predicted_noise / (1.0 - alpha_bar_t).sqrt()) / alpha_t.sqrt()

            if step > 0:
                variance = self._extract(self.posterior_variance, timesteps, current.shape)
                current = mean + variance.sqrt() * torch.randn_like(current)
            else:
                current = mean

        return current.clamp(-1.0, 1.0)
