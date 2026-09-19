"""Explicitly configured masked diffusion primitives, not a frozen study model."""
import torch
from torch import nn, Tensor


class SequenceDenoiser(nn.Module):
    """Small global sequence MLP; width/embedding size must be supplied by config."""

    def __init__(self, vocabulary_size: int, sequence_length: int, condition_dim: int,
                 *, width: int, embedding_dim: int):
        super().__init__()
        if min(vocabulary_size, sequence_length, condition_dim, width, embedding_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.sequence_length = sequence_length
        self.clean_states = vocabulary_size - 1  # MASK is last and never a clean target.
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(sequence_length * embedding_dim + condition_dim + 1, width), nn.SiLU(),
            nn.Linear(width, sequence_length * self.clean_states),
        )

    def forward(self, value: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        inputs = torch.cat((self.embedding(value).flatten(1), time[:, None], condition), dim=1)
        return self.network(inputs).reshape(len(value), self.sequence_length, self.clean_states)


class MaskedDiffusion:
    """Absorbing MASK corruption and categorical reverse unmasking.

    Schedule is cumulative mask probability from 0 to 1. EOS/PAD are treated
    identically to payload. No true length, grammar repair, or output truncation.
    """

    def __init__(self, mask_token: int, mask_probabilities, *, device: torch.device):
        probabilities = torch.as_tensor(mask_probabilities, dtype=torch.float32, device=device)
        if (mask_token < 2 or probabilities.ndim != 1 or len(probabilities) < 2
                or not torch.isfinite(probabilities).all()
                or probabilities[0] != 0 or probabilities[-1] != 1
                or not (probabilities[1:] > probabilities[:-1]).all()):
            raise ValueError("mask schedule must strictly increase from 0 to 1")
        self.probabilities = probabilities
        self.mask_token = mask_token
        self.steps = len(probabilities) - 1
        self.device = device

    def corrupt(self, clean: Tensor, index: Tensor, *, generator: torch.Generator):
        if clean.dtype != torch.long or ((clean < 0) | (clean >= self.mask_token)).any():
            raise ValueError("clean tokens must exclude MASK and unknown states")
        if ((index < 0) | (index > self.steps)).any():
            raise ValueError("timestep outside schedule")
        masked = torch.rand(clean.shape, device=self.device, generator=generator) < self.probabilities[index, None]
        return clean.masked_fill(masked, self.mask_token), masked

    def loss(self, model, clean: Tensor, condition: Tensor, *, generator: torch.Generator):
        index = torch.randint(1, self.steps + 1, (len(clean),), device=self.device, generator=generator)
        noisy, masked = self.corrupt(clean, index, generator=generator)
        logits = model(noisy, index.float() / self.steps, condition)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite categorical logits")
        # An empty corruption contributes zero; never reveal padding to force a mask.
        if not masked.any():
            return logits.sum() * 0
        losses = nn.functional.cross_entropy(logits.transpose(1, 2), clean, reduction="none")
        return ((losses * masked).sum(1) / masked.sum(1).clamp_min(1)).mean()

    @torch.no_grad()
    def sample(self, model, condition: Tensor, shape: tuple[int, ...], *,
               sampling_steps: int, generator: torch.Generator, temperature: float):
        if len(shape) != 1 or not 1 <= sampling_steps <= self.steps or not 0 < temperature < float("inf"):
            raise ValueError("invalid sequence shape, steps or temperature")
        times = torch.linspace(self.steps, 0, sampling_steps + 1, device=self.device).round().long()
        value = torch.full((len(condition), *shape), self.mask_token, dtype=torch.long, device=self.device)
        for current, previous in zip(times[:-1], times[1:]):
            time = torch.full((len(condition),), current.item() / self.steps, device=self.device)
            logits = model(value, time, condition)
            if logits.shape != (*value.shape, self.mask_token) or not torch.isfinite(logits).all():
                raise FloatingPointError("invalid categorical output")
            tokens = torch.multinomial((logits / temperature).softmax(-1).flatten(0, 1), 1,
                                       generator=generator).reshape_as(value)
            probability = 1 - self.probabilities[previous] / self.probabilities[current]
            reveal = (value == self.mask_token) & (torch.rand(value.shape, device=self.device, generator=generator) < probability)
            value = torch.where(reveal, tokens, value)
        return value
