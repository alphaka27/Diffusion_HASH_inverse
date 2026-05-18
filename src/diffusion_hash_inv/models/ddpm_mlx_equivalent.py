"""MLX U-Net equivalents for the Torch DDPM model family.

The existing ``conditional_diffusion_mlx`` module uses a compact flattened MLP
denoiser.  This module keeps the image-tensor U-Net structure used by the Torch
models so backend comparisons can share the same model family:

* base conditional DDPM -> :class:`MLXConditionalUNet`
* guided conditional DDPM -> :class:`MLXConditionalUNet` plus
  :class:`MLXNoisyImageClassifier`
* loop-conditioned DDPM -> :class:`MLXLoopConditionedUNet`
* unconditional DDPM -> :class:`MLXUnconditionalUNet`

MLX convolution layers use channels-last tensors, so all model inputs and
outputs are ``NHWC`` arrays.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn

GuidanceMode = Literal["classifier", "classifier-free"]
TemporalConditioningMode = Literal[
    "class",
    "loop-sinusoidal",
    "loop-structured",
    "loop-sequence",
]
UINT32_BYTE_COUNT = 4


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def timestep_embedding(t: mx.array, dim: int, max_period: int = 10_000) -> mx.array:
    """Sinusoidal timestep embedding matching the Torch implementation."""
    half = dim // 2
    if half == 0:
        return mx.zeros((t.shape[0], 0), dtype=mx.float32)
    scale = np.log(max_period) / max(half - 1, 1)
    freqs = mx.exp(-scale * mx.arange(half, dtype=mx.float32))
    args = t.astype(mx.float32)[:, None] * freqs[None, :]
    embedding = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate(
            [embedding, mx.zeros((embedding.shape[0], 1), dtype=embedding.dtype)],
            axis=-1,
        )
    return embedding


def nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


def _match_spatial(x: mx.array, target: mx.array) -> mx.array:
    """Nearest-resize ``x`` to ``target`` spatial size."""
    target_h, target_w = int(target.shape[1]), int(target.shape[2])
    height, width = int(x.shape[1]), int(x.shape[2])
    if height == target_h and width == target_w:
        return x
    row_idx = mx.minimum(
        (mx.arange(target_h, dtype=mx.float32) * (height / target_h)).astype(mx.int32),
        height - 1,
    )
    col_idx = mx.minimum(
        (mx.arange(target_w, dtype=mx.float32) * (width / target_w)).astype(mx.int32),
        width - 1,
    )
    return mx.take(mx.take(x, row_idx, axis=1), col_idx, axis=2)


def _valid_num_heads(dim: int, preferred: int = 4) -> int:
    for heads in range(preferred, 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class MLXConditionalResBlock(nn.Module):
    """Residual block conditioned by a timestep/label embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            _group_count(in_channels),
            in_channels,
            pytorch_compatible=True,
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            _group_count(out_channels),
            out_channels,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.emb_proj(nn.silu(emb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        skip = x if self.skip is None else self.skip(x)
        return h + skip


class MLXLoopSinusoidalConditioner(nn.Module):
    """MLX equivalent of ``LoopSinusoidalConditioner``."""

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.fc1 = nn.Linear(time_dim, time_dim * 4)
        self.fc2 = nn.Linear(time_dim * 4, time_dim)

    def __call__(self, loop_meta: mx.array) -> mx.array:
        loop_idx = loop_meta[:, 0].astype(mx.int32)
        emb = timestep_embedding(loop_idx, self.time_dim)
        return self.fc2(nn.silu(self.fc1(emb)))


class MLXLoopStructuredConditioner(nn.Module):
    """MLX equivalent of ``LoopStructuredConditioner``."""

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.idx_fc1 = nn.Linear(time_dim, time_dim * 4)
        self.idx_fc2 = nn.Linear(time_dim * 4, time_dim)
        self.start_fc1 = nn.Linear(1, time_dim)
        self.start_fc2 = nn.Linear(time_dim, time_dim)
        self.end_fc1 = nn.Linear(1, time_dim)
        self.end_fc2 = nn.Linear(time_dim, time_dim)

    def __call__(self, loop_meta: mx.array) -> mx.array:
        loop_idx = loop_meta[:, 0].astype(mx.int32)
        loop_start = loop_meta[:, 2:3]
        loop_end = loop_meta[:, 3:4]
        idx_emb = timestep_embedding(loop_idx, self.time_dim)
        idx_emb = self.idx_fc2(nn.silu(self.idx_fc1(idx_emb)))
        start_emb = self.start_fc2(nn.silu(self.start_fc1(loop_start)))
        end_emb = self.end_fc2(nn.silu(self.end_fc1(loop_end)))
        return idx_emb + start_emb + end_emb


class MLXLoopSequenceConditioner(nn.Module):
    """MLX equivalent of ``LoopSequenceConditioner``."""

    def __init__(
        self,
        max_loop_count: int,
        time_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.max_loop_count = max_loop_count
        self.time_dim = time_dim
        self.loop_tokens = nn.Embedding(max_loop_count, time_dim)
        self.transformer = nn.TransformerEncoder(
            num_layers,
            time_dim,
            _valid_num_heads(time_dim, num_heads),
            mlp_dims=time_dim * 4,
            norm_first=True,
        )
        self.proj = nn.Linear(time_dim, time_dim)

    def __call__(self, loop_meta: mx.array) -> mx.array:
        loop_idx = loop_meta[:, 0].astype(mx.int32)
        ctx = self.transformer(self.loop_tokens.weight[None, :, :], None)[0]
        return nn.silu(self.proj(ctx[loop_idx]))


class MLXConditionalUNet(nn.Module):
    """MLX equivalent of the Torch ``ConditionalUNet``.

    Inputs and outputs use NHWC layout.
    """

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
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.in_channels = in_channels
        self.num_conditions = num_conditions
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.temporal_conditioning = temporal_conditioning
        self.max_loop_count = max_loop_count

        self.time_fc1 = nn.Linear(time_dim, time_dim * 4)
        self.time_fc2 = nn.Linear(time_dim * 4, time_dim)
        if temporal_conditioning == "class":
            self.condition_embedding: nn.Module = nn.Embedding(num_conditions, time_dim)
        elif temporal_conditioning == "loop-sinusoidal":
            self.condition_embedding = MLXLoopSinusoidalConditioner(time_dim)
        elif temporal_conditioning == "loop-structured":
            self.condition_embedding = MLXLoopStructuredConditioner(time_dim)
        elif temporal_conditioning == "loop-sequence":
            self.condition_embedding = MLXLoopSequenceConditioner(max_loop_count, time_dim)
        else:
            raise ValueError(f"Unsupported temporal_conditioning: {temporal_conditioning!r}")

        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = MLXConditionalResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down2 = MLXConditionalResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid = MLXConditionalResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up1 = MLXConditionalResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up2 = MLXConditionalResBlock(base_channels * 2, base_channels, time_dim)
        self.out_norm = nn.GroupNorm(
            _group_count(base_channels),
            base_channels,
            pytorch_compatible=True,
        )
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def _conditioning_embedding(
        self,
        labels: mx.array,
        loop_meta: mx.array | None,
    ) -> mx.array:
        if self.temporal_conditioning == "class":
            return self.condition_embedding(labels.astype(mx.int32))
        if loop_meta is None:
            raise ValueError(
                f"loop_meta is required for temporal_conditioning={self.temporal_conditioning!r}"
            )
        return self.condition_embedding(loop_meta)

    def _embedding(
        self,
        timesteps: mx.array,
        labels: mx.array,
        loop_meta: mx.array | None = None,
    ) -> mx.array:
        t_emb = timestep_embedding(timesteps, self.time_dim)
        t_emb = self.time_fc2(nn.silu(self.time_fc1(t_emb)))
        return t_emb + self._conditioning_embedding(labels, loop_meta)

    def __call__(
        self,
        x: mx.array,
        timesteps: mx.array,
        labels: mx.array,
        loop_meta: mx.array | None = None,
    ) -> mx.array:
        emb = self._embedding(timesteps, labels, loop_meta)
        x0 = self.input(x)
        d1 = self.down1(x0, emb)
        d2_in = self.downsample1(d1)
        d2 = self.down2(d2_in, emb)
        mid_in = self.downsample2(d2)
        mid = self.mid(mid_in, emb)
        u1 = _match_spatial(self.upsample1(mid), d2)
        u1 = self.up1(mx.concatenate([u1, d2], axis=-1), emb)
        u2 = _match_spatial(self.upsample2(u1), d1)
        u2 = self.up2(mx.concatenate([u2, d1], axis=-1), emb)
        return self.out_conv(nn.silu(self.out_norm(u2)))


class MLXUnconditionalUNet(nn.Module):
    """MLX equivalent of the Torch ``UnconditionalUNet``."""

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

        self.time_fc1 = nn.Linear(time_dim, time_dim * 4)
        self.time_fc2 = nn.Linear(time_dim * 4, time_dim)
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = MLXConditionalResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down2 = MLXConditionalResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid = MLXConditionalResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up1 = MLXConditionalResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up2 = MLXConditionalResBlock(base_channels * 2, base_channels, time_dim)
        self.out_norm = nn.GroupNorm(
            _group_count(base_channels),
            base_channels,
            pytorch_compatible=True,
        )
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array, timesteps: mx.array) -> mx.array:
        emb = timestep_embedding(timesteps, self.time_dim)
        emb = self.time_fc2(nn.silu(self.time_fc1(emb)))
        x0 = self.input(x)
        d1 = self.down1(x0, emb)
        d2_in = self.downsample1(d1)
        d2 = self.down2(d2_in, emb)
        mid_in = self.downsample2(d2)
        mid = self.mid(mid_in, emb)
        u1 = _match_spatial(self.upsample1(mid), d2)
        u1 = self.up1(mx.concatenate([u1, d2], axis=-1), emb)
        u2 = _match_spatial(self.upsample2(u1), d1)
        u2 = self.up2(mx.concatenate([u2, d1], axis=-1), emb)
        return self.out_conv(nn.silu(self.out_norm(u2)))


class MLXLoopConditionedUNet(nn.Module):
    """MLX equivalent of the Torch ``LoopConditionedUNet``."""

    def __init__(
        self,
        in_channels: int,
        condition_shape: tuple[int, int] = (66, 4),
        base_channels: int = 64,
        time_dim: int = 256,
        diffusion_timesteps: int | None = None,
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
        self.diffusion_timesteps = diffusion_timesteps

        self.time_fc1 = nn.Linear(time_dim, time_dim * 4)
        self.time_fc2 = nn.Linear(time_dim * 4, time_dim)
        self.state_fc1 = nn.Linear(self.condition_shape[1], time_dim * 2)
        self.state_fc2 = nn.Linear(time_dim * 2, time_dim)
        self.state_pos_fc1 = nn.Linear(time_dim, time_dim * 2)
        self.state_pos_fc2 = nn.Linear(time_dim * 2, time_dim)

        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = MLXConditionalResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down2 = MLXConditionalResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid = MLXConditionalResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up1 = MLXConditionalResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up2 = MLXConditionalResBlock(base_channels * 2, base_channels, time_dim)
        self.out_norm = nn.GroupNorm(
            _group_count(base_channels),
            base_channels,
            pytorch_compatible=True,
        )
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def _state_indices(self, timesteps: mx.array) -> mx.array:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be one-dimensional, got shape {timesteps.shape}")
        state_count = self.condition_shape[0]
        inferred_timesteps = state_count * self.condition_shape[1] * UINT32_BYTE_COUNT
        total_timesteps = (
            inferred_timesteps
            if self.diffusion_timesteps is None
            else int(self.diffusion_timesteps)
        )
        if total_timesteps <= 0:
            raise ValueError("diffusion_timesteps must be positive")
        indices = (timesteps.astype(mx.int64) * state_count) // total_timesteps
        return mx.minimum(indices, state_count - 1).astype(mx.int32)

    def _embedding(self, timesteps: mx.array, conditions: mx.array) -> mx.array:
        expected = (conditions.shape[0], *self.condition_shape)
        if tuple(conditions.shape) != expected:
            raise ValueError(f"conditions must have shape {expected}, got {conditions.shape}")
        t_emb = timestep_embedding(timesteps, self.time_dim)
        t_emb = self.time_fc2(nn.silu(self.time_fc1(t_emb)))
        state_indices = self._state_indices(timesteps)
        batch_indices = mx.arange(conditions.shape[0], dtype=mx.int32)
        state_values = conditions[batch_indices, state_indices, :]
        state_emb = self.state_fc2(nn.silu(self.state_fc1(state_values)))
        pos_emb = timestep_embedding(state_indices, self.time_dim)
        pos_emb = self.state_pos_fc2(nn.silu(self.state_pos_fc1(pos_emb)))
        return t_emb + state_emb + pos_emb

    def __call__(self, x: mx.array, timesteps: mx.array, conditions: mx.array) -> mx.array:
        emb = self._embedding(timesteps, conditions)
        x0 = self.input(x)
        d1 = self.down1(x0, emb)
        d2_in = self.downsample1(d1)
        d2 = self.down2(d2_in, emb)
        mid_in = self.downsample2(d2)
        mid = self.mid(mid_in, emb)
        u1 = _match_spatial(self.upsample1(mid), d2)
        u1 = self.up1(mx.concatenate([u1, d2], axis=-1), emb)
        u2 = _match_spatial(self.upsample2(u1), d1)
        u2 = self.up2(mx.concatenate([u2, d1], axis=-1), emb)
        return self.out_conv(nn.silu(self.out_norm(u2)))


class MLXNoisyImageClassifier(nn.Module):
    """MLX equivalent of the noisy ``x_t`` classifier used for guidance."""

    def __init__(
        self,
        in_channels: int,
        num_conditions: int,
        base_channels: int = 32,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_conditions = num_conditions
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.time_fc1 = nn.Linear(time_dim, time_dim * 2)
        self.time_fc2 = nn.Linear(time_dim * 2, time_dim)
        self.time_projection = nn.Linear(time_dim, base_channels)
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.block1_norm = nn.GroupNorm(
            _group_count(base_channels),
            base_channels,
            pytorch_compatible=True,
        )
        self.block1_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.down1_norm = nn.GroupNorm(
            _group_count(base_channels),
            base_channels,
            pytorch_compatible=True,
        )
        self.down1_conv = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down2_norm = nn.GroupNorm(
            _group_count(base_channels * 2),
            base_channels * 2,
            pytorch_compatible=True,
        )
        self.down2_conv = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.head_norm = nn.GroupNorm(
            _group_count(base_channels * 4),
            base_channels * 4,
            pytorch_compatible=True,
        )
        self.head = nn.Linear(base_channels * 4, num_conditions)

    def __call__(self, x: mx.array, timesteps: mx.array) -> mx.array:
        t_emb = timestep_embedding(timesteps, self.time_dim)
        t_emb = self.time_fc2(nn.silu(self.time_fc1(t_emb)))
        h = self.input(x) + self.time_projection(t_emb)[:, None, None, :]
        h = self.block1_conv(nn.silu(self.block1_norm(h)))
        h = self.down1_conv(nn.silu(self.down1_norm(h)))
        h = self.down2_conv(nn.silu(self.down2_norm(h)))
        h = nn.silu(self.head_norm(h))
        h = mx.mean(h, axis=(1, 2))
        return self.head(h)


class MLXImageDDPMScheduler:
    """DDPM scheduler for MLX image tensors in NHWC layout."""

    def __init__(
        self,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        betas: np.ndarray | list[float] | None = None,
    ) -> None:
        if betas is None:
            if timesteps <= 0:
                raise ValueError("timesteps must be positive")
            self.timesteps = int(timesteps)
            self.betas = mx.linspace(beta_start, beta_end, self.timesteps, dtype=mx.float32)
        else:
            beta_values = np.asarray(betas, dtype=np.float32)
            if beta_values.ndim != 1 or beta_values.size == 0:
                raise ValueError("betas must be a non-empty 1D array")
            if not np.isfinite(beta_values).all():
                raise ValueError("betas contains non-finite values")
            if (beta_values <= 0).any() or (beta_values >= 1).any():
                raise ValueError("betas must be in the open interval (0, 1)")
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
        values = mx.take(coefficients, t.astype(mx.int32), axis=0)
        return values.reshape((t.shape[0],) + (1,) * (target_ndim - 1))

    def sample_timesteps(self, batch_size: int) -> mx.array:
        return mx.random.randint(0, self.timesteps, shape=(batch_size,), dtype=mx.int32)

    def q_sample(self, x0: mx.array, t: mx.array, noise: mx.array | None = None) -> mx.array:
        if noise is None:
            noise = mx.random.normal(x0.shape, dtype=mx.float32)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.ndim)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.ndim)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def posterior_mean(self, x: mx.array, t: mx.array, pred_noise: mx.array) -> mx.array:
        beta_t = self._extract(self.betas, t, x.ndim)
        alpha_t = self._extract(self.alphas, t, x.ndim)
        alpha_bar_t = self._extract(self.alpha_bars, t, x.ndim)
        return (1.0 / mx.sqrt(alpha_t)) * (
            x - (beta_t / mx.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

    def posterior_variance_at(self, x: mx.array, t: mx.array) -> mx.array:
        return self._extract(self.posterior_variance, t, x.ndim)

    def p_sample_unconditional(
        self,
        model: MLXUnconditionalUNet,
        x: mx.array,
        step: int,
    ) -> mx.array:
        t = mx.full((x.shape[0],), step, dtype=mx.int32)
        pred_noise = model(x, t)
        mean = self.posterior_mean(x, t, pred_noise)
        if step == 0:
            return mean
        variance = self.posterior_variance_at(x, t)
        return mean + mx.sqrt(variance) * mx.random.normal(x.shape, dtype=mx.float32)

    def p_sample_conditional(
        self,
        model: MLXConditionalUNet,
        x: mx.array,
        step: int,
        labels: mx.array,
        loop_meta: mx.array | None = None,
    ) -> mx.array:
        t = mx.full((x.shape[0],), step, dtype=mx.int32)
        pred_noise = model(x, t, labels, loop_meta)
        mean = self.posterior_mean(x, t, pred_noise)
        if step == 0:
            return mean
        variance = self.posterior_variance_at(x, t)
        return mean + mx.sqrt(variance) * mx.random.normal(x.shape, dtype=mx.float32)

    def p_sample_loop_conditioned(
        self,
        model: MLXLoopConditionedUNet,
        x: mx.array,
        step: int,
        conditions: mx.array,
    ) -> mx.array:
        t = mx.full((x.shape[0],), step, dtype=mx.int32)
        pred_noise = model(x, t, conditions)
        mean = self.posterior_mean(x, t, pred_noise)
        if step == 0:
            return mean
        variance = self.posterior_variance_at(x, t)
        return mean + mx.sqrt(variance) * mx.random.normal(x.shape, dtype=mx.float32)


def conditional_diffusion_loss(
    model: MLXConditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    x0: mx.array,
    labels: mx.array,
    loop_meta: mx.array | None = None,
) -> mx.array:
    t = scheduler.sample_timesteps(x0.shape[0])
    noise = mx.random.normal(x0.shape, dtype=mx.float32)
    xt = scheduler.q_sample(x0, t, noise)
    pred_noise = model(xt, t, labels, loop_meta)
    return mx.mean((pred_noise - noise) ** 2)


def apply_condition_dropout(
    labels: mx.array,
    null_label: int,
    dropout: float,
) -> mx.array:
    """Replace labels with ``null_label`` with probability ``dropout``."""
    if dropout < 0.0 or dropout > 1.0:
        raise ValueError("condition_dropout must be in the interval [0, 1]")
    if dropout == 0.0:
        return labels
    mask = mx.random.uniform(shape=labels.shape) < dropout
    null_labels = mx.full(labels.shape, int(null_label), dtype=labels.dtype)
    return mx.where(mask, null_labels, labels)


def classifier_free_diffusion_loss(
    model: MLXConditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    x0: mx.array,
    labels: mx.array,
    *,
    null_label: int,
    condition_dropout: float,
    loop_meta: mx.array | None = None,
) -> mx.array:
    train_labels = apply_condition_dropout(labels, null_label, condition_dropout)
    return conditional_diffusion_loss(model, scheduler, x0, train_labels, loop_meta)


def classifier_guided_denoiser_loss(
    model: MLXConditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    x0: mx.array,
    labels: mx.array,
    loop_meta: mx.array | None = None,
) -> mx.array:
    model_labels = mx.zeros(labels.shape, dtype=mx.int32)
    return conditional_diffusion_loss(model, scheduler, x0, model_labels, loop_meta)


def noisy_classifier_loss(
    classifier: MLXNoisyImageClassifier,
    scheduler: MLXImageDDPMScheduler,
    x0: mx.array,
    labels: mx.array,
) -> mx.array:
    t = scheduler.sample_timesteps(x0.shape[0])
    noise = mx.random.normal(x0.shape, dtype=mx.float32)
    xt = scheduler.q_sample(x0, t, noise)
    logits = classifier(xt, t)
    return nn.losses.cross_entropy(logits, labels.astype(mx.int32), reduction="mean")


def make_conditional_train_step(
    model: MLXConditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    optimizer,
):
    loss_and_grad_fn = nn.value_and_grad(
        model,
        lambda m, x0, labels, loop_meta: conditional_diffusion_loss(
            m,
            scheduler,
            x0,
            labels,
            loop_meta,
        ),
    )

    def train_step(
        x0: mx.array,
        labels: mx.array,
        loop_meta: mx.array | None = None,
    ) -> mx.array:
        loss, grads = loss_and_grad_fn(model, x0, labels, loop_meta)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    return train_step


def make_classifier_free_train_step(
    model: MLXConditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    optimizer,
    *,
    null_label: int,
    condition_dropout: float,
):
    loss_and_grad_fn = nn.value_and_grad(
        model,
        lambda m, x0, labels, loop_meta: classifier_free_diffusion_loss(
            m,
            scheduler,
            x0,
            labels,
            null_label=null_label,
            condition_dropout=condition_dropout,
            loop_meta=loop_meta,
        ),
    )

    def train_step(
        x0: mx.array,
        labels: mx.array,
        loop_meta: mx.array | None = None,
    ) -> mx.array:
        loss, grads = loss_and_grad_fn(model, x0, labels, loop_meta)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    return train_step


def make_classifier_guided_denoiser_train_step(
    model: MLXConditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    optimizer,
):
    loss_and_grad_fn = nn.value_and_grad(
        model,
        lambda m, x0, labels, loop_meta: classifier_guided_denoiser_loss(
            m,
            scheduler,
            x0,
            labels,
            loop_meta,
        ),
    )

    def train_step(
        x0: mx.array,
        labels: mx.array,
        loop_meta: mx.array | None = None,
    ) -> mx.array:
        loss, grads = loss_and_grad_fn(model, x0, labels, loop_meta)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    return train_step


def make_noisy_classifier_train_step(
    classifier: MLXNoisyImageClassifier,
    scheduler: MLXImageDDPMScheduler,
    optimizer,
):
    loss_and_grad_fn = nn.value_and_grad(
        classifier,
        lambda m, x0, labels: noisy_classifier_loss(m, scheduler, x0, labels),
    )

    def train_step(x0: mx.array, labels: mx.array) -> mx.array:
        loss, grads = loss_and_grad_fn(classifier, x0, labels)
        optimizer.update(classifier, grads)
        mx.eval(classifier.parameters(), optimizer.state)
        return loss

    return train_step


def unconditional_diffusion_loss(
    model: MLXUnconditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    x0: mx.array,
) -> mx.array:
    t = scheduler.sample_timesteps(x0.shape[0])
    noise = mx.random.normal(x0.shape, dtype=mx.float32)
    xt = scheduler.q_sample(x0, t, noise)
    pred_noise = model(xt, t)
    return mx.mean((pred_noise - noise) ** 2)


def make_unconditional_train_step(
    model: MLXUnconditionalUNet,
    scheduler: MLXImageDDPMScheduler,
    optimizer,
):
    loss_and_grad_fn = nn.value_and_grad(
        model,
        lambda m, x0: unconditional_diffusion_loss(m, scheduler, x0),
    )

    def train_step(x0: mx.array) -> mx.array:
        loss, grads = loss_and_grad_fn(model, x0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    return train_step


def loop_conditioned_diffusion_loss(
    model: MLXLoopConditionedUNet,
    scheduler: MLXImageDDPMScheduler,
    x0: mx.array,
    conditions: mx.array,
) -> mx.array:
    t = scheduler.sample_timesteps(x0.shape[0])
    noise = mx.random.normal(x0.shape, dtype=mx.float32)
    xt = scheduler.q_sample(x0, t, noise)
    pred_noise = model(xt, t, conditions)
    return mx.mean((pred_noise - noise) ** 2)


def make_loop_conditioned_train_step(
    model: MLXLoopConditionedUNet,
    scheduler: MLXImageDDPMScheduler,
    optimizer,
):
    loss_and_grad_fn = nn.value_and_grad(
        model,
        lambda m, x0, conditions: loop_conditioned_diffusion_loss(
            m,
            scheduler,
            x0,
            conditions,
        ),
    )

    def train_step(x0: mx.array, conditions: mx.array) -> mx.array:
        loss, grads = loss_and_grad_fn(model, x0, conditions)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    return train_step


def p_sample_classifier_free_guidance(
    scheduler: MLXImageDDPMScheduler,
    model: MLXConditionalUNet,
    x: mx.array,
    step: int,
    labels: mx.array,
    *,
    loop_meta: mx.array | None = None,
    null_label: int,
    guidance_scale: float,
) -> mx.array:
    t = mx.full((x.shape[0],), step, dtype=mx.int32)
    null_labels = mx.full(labels.shape, int(null_label), dtype=mx.int32)
    eps_uncond = model(x, t, null_labels, loop_meta)
    eps_cond = model(x, t, labels, loop_meta)
    pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    mean = scheduler.posterior_mean(x, t, pred_noise)
    if step == 0:
        return mean
    variance = scheduler.posterior_variance_at(x, t)
    return mean + mx.sqrt(variance) * mx.random.normal(x.shape, dtype=mx.float32)


def _classifier_log_prob_gradient(
    classifier: MLXNoisyImageClassifier,
    x: mx.array,
    timesteps: mx.array,
    labels: mx.array,
) -> mx.array:
    def selected_log_prob(x_in: mx.array) -> mx.array:
        logits = classifier(x_in, timesteps)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        one_hot = mx.take(mx.eye(classifier.num_conditions), labels.astype(mx.int32), axis=0)
        return mx.sum(log_probs * one_hot)

    return mx.grad(selected_log_prob)(x)


def p_sample_classifier_guidance(
    scheduler: MLXImageDDPMScheduler,
    model: MLXConditionalUNet,
    classifier: MLXNoisyImageClassifier,
    x: mx.array,
    step: int,
    labels: mx.array,
    *,
    loop_meta: mx.array | None = None,
    guidance_scale: float,
) -> mx.array:
    t = mx.full((x.shape[0],), step, dtype=mx.int32)
    model_labels = mx.zeros(labels.shape, dtype=mx.int32)
    pred_noise = model(x, t, model_labels, loop_meta)
    mean = scheduler.posterior_mean(x, t, pred_noise)
    variance = scheduler.posterior_variance_at(x, t)
    if guidance_scale != 0.0:
        grad = _classifier_log_prob_gradient(classifier, x, t, labels)
        mean = mean + variance * guidance_scale * grad
    if step == 0:
        return mean
    return mean + mx.sqrt(variance) * mx.random.normal(x.shape, dtype=mx.float32)


def sample_unconditional(
    scheduler: MLXImageDDPMScheduler,
    model: MLXUnconditionalUNet,
    shape: Sequence[int],
) -> mx.array:
    x = mx.random.normal(tuple(shape), dtype=mx.float32)
    for step in reversed(range(scheduler.timesteps)):
        x = scheduler.p_sample_unconditional(model, x, step)
    return mx.clip(x, -1.0, 1.0)


def sample_conditional(
    scheduler: MLXImageDDPMScheduler,
    model: MLXConditionalUNet,
    labels: mx.array,
    image_shape: tuple[int, int, int],
    *,
    loop_meta: mx.array | None = None,
) -> mx.array:
    x = mx.random.normal((labels.shape[0], *image_shape), dtype=mx.float32)
    for step in reversed(range(scheduler.timesteps)):
        x = scheduler.p_sample_conditional(model, x, step, labels, loop_meta)
    return mx.clip(x, -1.0, 1.0)


def sample_classifier_free_guidance(
    scheduler: MLXImageDDPMScheduler,
    model: MLXConditionalUNet,
    labels: mx.array,
    image_shape: tuple[int, int, int],
    *,
    loop_meta: mx.array | None = None,
    null_label: int,
    guidance_scale: float,
) -> mx.array:
    x = mx.random.normal((labels.shape[0], *image_shape), dtype=mx.float32)
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
    return mx.clip(x, -1.0, 1.0)


def sample_classifier_guidance(
    scheduler: MLXImageDDPMScheduler,
    model: MLXConditionalUNet,
    classifier: MLXNoisyImageClassifier,
    labels: mx.array,
    image_shape: tuple[int, int, int],
    *,
    loop_meta: mx.array | None = None,
    guidance_scale: float,
) -> mx.array:
    x = mx.random.normal((labels.shape[0], *image_shape), dtype=mx.float32)
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
    return mx.clip(x, -1.0, 1.0)


def sample_loop_conditioned(
    scheduler: MLXImageDDPMScheduler,
    model: MLXLoopConditionedUNet,
    conditions: mx.array,
    image_shape: tuple[int, int, int],
) -> mx.array:
    x = mx.random.normal((conditions.shape[0], *image_shape), dtype=mx.float32)
    for step in reversed(range(scheduler.timesteps)):
        x = scheduler.p_sample_loop_conditioned(model, x, step, conditions)
    return mx.clip(x, -1.0, 1.0)
