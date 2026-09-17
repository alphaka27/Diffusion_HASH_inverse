"""Small conditional diffusion components with no dependencies beyond PyTorch."""

from __future__ import annotations

from math import prod
from typing import Literal

import torch
from torch import Tensor, nn


class ImageUNet(nn.Module):
    """A compact conditional U-Net shared by BGV and CGGE pixel experiments."""

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        width: int = 32,
        *,
        condition_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if condition_shape is not None and (condition_shape[0] != channels or prod(condition_shape) != condition_dim):
            raise ValueError("spatial condition must match the image shape")
        self.condition_shape = condition_shape
        self.input = nn.Conv2d(channels * (2 if condition_shape else 1), width, 3, padding=1)
        self.down = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.middle = nn.Conv2d(width * 2, width * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.output = nn.Conv2d(width * 2, channels, 3, padding=1)
        self.condition = nn.Sequential(
            nn.Linear(1 if condition_shape else condition_dim + 1, width * 3),
            nn.SiLU(),
            nn.Linear(width * 3, width * 3),
        )
        self.activation = nn.SiLU()

    def forward(self, value: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if self.condition_shape:
            value = torch.cat((value, condition.reshape((len(condition), *self.condition_shape))), dim=1)
            embedding = self.condition(time[:, None])
        else:
            embedding = self.condition(torch.cat((condition, time[:, None]), dim=1))
        skip = self.activation(self.input(value) + embedding[:, : self.input.out_channels, None, None])
        hidden = self.activation(self.down(skip) + embedding[:, self.input.out_channels :, None, None])
        hidden = self.activation(self.middle(hidden))
        hidden = self.activation(self.up(hidden))
        return self.output(torch.cat((hidden, skip), dim=1))


class BitDenoiser(nn.Module):
    """A conditional MLP denoiser for flat direct-bit records."""

    def __init__(self, record_bits: int, condition_dim: int, width: int = 256, *, aligned_condition: bool = False) -> None:
        super().__init__()
        if aligned_condition and record_bits != condition_dim:
            raise ValueError("aligned condition must match the record width")
        point_width = min(width, 32)
        self.aligned_condition = (
            nn.Sequential(nn.Linear(3, point_width), nn.SiLU(), nn.Linear(point_width, 1))
            if aligned_condition
            else None
        )
        self.network = (
            None
            if aligned_condition
            else nn.Sequential(
                nn.Linear(record_bits + condition_dim + 1, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, record_bits),
            )
        )

    def forward(self, value: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if self.aligned_condition is not None:
            flat = value.flatten(1)
            inputs = torch.stack((flat, condition, time[:, None].expand_as(flat)), dim=-1)
            return self.aligned_condition(inputs).reshape_as(value)
        return self.network(torch.cat((value.flatten(1), condition, time[:, None]), dim=1)).reshape_as(value)


class DirectPredictor(nn.Module):
    """Non-diffusion conditional record predictor used as a control baseline."""

    def __init__(self, condition_dim: int, output_size: int, width: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, output_size),
        )

    def forward(self, condition: Tensor, shape: tuple[int, ...]) -> Tensor:
        return self.network(condition).reshape((len(condition), *shape))


class GaussianDiffusion:
    """DDIM-style sampler with epsilon- or clean-sample prediction."""

    def __init__(
        self,
        steps: int,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: Literal["epsilon", "sample"] = "epsilon",
        device: torch.device,
    ) -> None:
        if steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("invalid diffusion schedule")
        if prediction_type not in {"epsilon", "sample"}:
            raise ValueError("prediction_type must be epsilon or sample")
        self.steps = steps
        self.device = device
        self.prediction_type = prediction_type
        beta = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alpha_bar = torch.cumprod(1 - beta, dim=0)

    def add_noise(self, clean: Tensor, noise: Tensor, index: Tensor) -> Tensor:
        """Apply q(x_t | x_0) with an explicit noise tensor for diagnostics."""
        alpha = self.alpha_bar[index].reshape((len(clean),) + (1,) * (clean.ndim - 1))
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def predicted_clean(self, model_output: Tensor, noisy: Tensor, index: Tensor) -> Tensor:
        """Convert the configured model parameterization to an x_0 estimate."""
        if self.prediction_type == "sample":
            return model_output
        alpha = self.alpha_bar[index].reshape((len(noisy),) + (1,) * (noisy.ndim - 1))
        return (noisy - (1 - alpha).sqrt() * model_output) / alpha.sqrt()

    def loss(self, model: nn.Module, clean: Tensor, condition: Tensor, *, generator: torch.Generator) -> Tensor:
        index = torch.randint(self.steps, (len(clean),), device=self.device, generator=generator)
        noise = torch.randn(clean.shape, device=self.device, generator=generator)
        noisy = self.add_noise(clean, noise, index)
        predicted = model(noisy, index.float() / (self.steps - 1), condition)
        return nn.functional.mse_loss(predicted, noise if self.prediction_type == "epsilon" else clean)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: Tensor,
        shape: tuple[int, ...],
        *,
        sampling_steps: int,
        generator: torch.Generator,
    ) -> Tensor:
        if not 1 <= sampling_steps <= self.steps:
            raise ValueError("sampling_steps must be within the diffusion schedule")
        times = torch.linspace(self.steps - 1, 0, sampling_steps, device=self.device).round().long().unique_consecutive()
        value = torch.randn((len(condition), *shape), device=self.device, generator=generator)
        for position, index in enumerate(times):
            alpha = self.alpha_bar[index]
            time = torch.full((len(condition),), index.item() / (self.steps - 1), device=self.device)
            output = model(value, time, condition)
            clean = self.predicted_clean(output, value, index.repeat(len(condition)))
            if self.prediction_type == "sample":
                clean = clean.clamp(-1, 1)
                noise = (value - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            else:
                noise = output
            previous_alpha = self.alpha_bar[times[position + 1]] if position + 1 < len(times) else torch.tensor(1.0, device=self.device)
            value = previous_alpha.sqrt() * clean + (1 - previous_alpha).sqrt() * noise
        return value.clamp(-1, 1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["BitDenoiser", "DirectPredictor", "GaussianDiffusion", "ImageUNet", "parameter_count"]
