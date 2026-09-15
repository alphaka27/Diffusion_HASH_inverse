"""Small conditional diffusion components with no dependencies beyond PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ImageUNet(nn.Module):
    """A compact conditional U-Net shared by BGV and CGGE pixel experiments."""

    def __init__(self, channels: int, condition_dim: int, width: int = 32) -> None:
        super().__init__()
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.down = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.middle = nn.Conv2d(width * 2, width * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.output = nn.Conv2d(width * 2, channels, 3, padding=1)
        self.condition = nn.Sequential(nn.Linear(condition_dim + 1, width * 3), nn.SiLU(), nn.Linear(width * 3, width * 3))
        self.activation = nn.SiLU()

    def forward(self, value: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        embedding = self.condition(torch.cat((condition, time[:, None]), dim=1))
        skip = self.activation(self.input(value) + embedding[:, : self.input.out_channels, None, None])
        hidden = self.activation(self.down(skip) + embedding[:, self.input.out_channels :, None, None])
        hidden = self.activation(self.middle(hidden))
        hidden = self.activation(self.up(hidden))
        return self.output(torch.cat((hidden, skip), dim=1))


class BitDenoiser(nn.Module):
    """A conditional MLP denoiser for flat direct-bit records."""

    def __init__(self, record_bits: int, condition_dim: int, width: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(record_bits + condition_dim + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, record_bits),
        )

    def forward(self, value: Tensor, time: Tensor, condition: Tensor) -> Tensor:
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
    """DDIM-style sampler and noise-prediction loss for tensors in [-1, 1]."""

    def __init__(self, steps: int, *, beta_start: float = 1e-4, beta_end: float = 0.02, device: torch.device) -> None:
        if steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("invalid diffusion schedule")
        self.steps = steps
        self.device = device
        beta = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alpha_bar = torch.cumprod(1 - beta, dim=0)

    def loss(self, model: nn.Module, clean: Tensor, condition: Tensor, *, generator: torch.Generator) -> Tensor:
        index = torch.randint(self.steps, (len(clean),), device=self.device, generator=generator)
        noise = torch.randn(clean.shape, device=self.device, generator=generator)
        alpha = self.alpha_bar[index].reshape((len(clean),) + (1,) * (clean.ndim - 1))
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        predicted = model(noisy, index.float() / (self.steps - 1), condition)
        return nn.functional.mse_loss(predicted, noise)

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
            noise = model(value, time, condition)
            clean = (value - (1 - alpha).sqrt() * noise) / alpha.sqrt()
            previous_alpha = self.alpha_bar[times[position + 1]] if position + 1 < len(times) else torch.tensor(1.0, device=self.device)
            value = previous_alpha.sqrt() * clean + (1 - previous_alpha).sqrt() * noise
        return value.clamp(-1, 1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["BitDenoiser", "DirectPredictor", "GaussianDiffusion", "ImageUNet", "parameter_count"]
