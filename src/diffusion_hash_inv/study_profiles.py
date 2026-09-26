"""The five registered v3.1 model profiles, built from existing primitives."""
import torch
from torch import nn

from .discrete import MaskedDiffusion, SequenceDenoiser
from .models import GaussianDiffusion, ImageUNet


class CoordinateImageUNet(ImageUNet):
    """Append public, fixed coordinates only to the denoiser's input."""

    def __init__(self, shape, width):
        super().__init__(2, 12, width)
        self.input = nn.Conv2d(4, width, 3, padding=1)
        height, width_pixels = shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, height),
                              torch.linspace(-1, 1, width_pixels), indexing="ij")
        self.register_buffer("coordinates", torch.stack((x, y))[None], persistent=False)

    def forward(self, value, time, condition):
        if value.shape[1:] != (2, *self.coordinates.shape[-2:]):
            raise ValueError("coordinate profile expects the original two-channel image")
        return super().forward(torch.cat((value, self.coordinates.expand(len(value), -1, -1, -1)), 1),
                               time, condition)


class LengthSequenceDenoiser(SequenceDenoiser):
    def __init__(self, vocabulary_size, *, width, embedding_dim):
        super().__init__(vocabulary_size, 32, 13, width=width, embedding_dim=embedding_dim)
        self.length_head = nn.Linear(12, 28)


class LengthMaskedDiffusion(MaskedDiffusion):
    """Learn length, then diffuse payload with intrinsic EOS/PAD context."""

    def lengths(self, clean):
        if clean.dtype != torch.long or clean.ndim != 2 or clean.shape[1] != 32:
            raise ValueError("D1 requires encoded 32-position training sequences")
        eos, pad = self.mask_token - 2, self.mask_token - 1
        locations = clean == eos
        if not (locations.sum(1) == 1).all():
            raise ValueError("D1 training records need exactly one EOS")
        lengths = locations.long().argmax(1)
        positions = torch.arange(32, device=clean.device)[None]
        if (((lengths < 4) | (lengths > 31)).any()
                or ((positions < lengths[:, None]) & ((clean < 0) | (clean >= eos))).any()
                or ((positions > lengths[:, None]) & (clean != pad)).any()):
            raise ValueError("D1 training record has invalid length/payload/padding")
        return lengths

    @staticmethod
    def payload_condition(condition, lengths):
        if condition.ndim != 2 or condition.shape[1] != 12:
            raise ValueError("D1 external condition contains only twelve public bits")
        return torch.cat((condition, lengths[:, None].to(condition.dtype) / 31), 1)

    def losses(self, model, clean, condition, times, masks):
        lengths = self.lengths(clean)
        masks = masks & (torch.arange(32, device=clean.device)[None] < lengths[:, None])
        logits = model(clean.masked_fill(masks, self.mask_token), times,
                       self.payload_condition(condition, lengths))[..., :self.mask_token - 2]
        length_logits = model.length_head(condition)
        if not torch.isfinite(logits).all() or not torch.isfinite(length_logits).all():
            raise FloatingPointError("non-finite D1 logits")
        # Non-payload positions never contribute, including when no token is masked.
        payload = nn.functional.cross_entropy(logits.transpose(1, 2),
                                              clean.clamp_max(self.mask_token - 3), reduction="none")
        return (nn.functional.cross_entropy(length_logits, lengths - 4, reduction="none")
                + (payload * masks).sum(1) / masks.sum(1).clamp_min(1))

    def loss(self, model, clean, condition, *, generator):
        times = torch.rand((len(clean),), device=self.device, generator=generator)
        masks = torch.rand(clean.shape, device=self.device, generator=generator) < times[:, None]
        return self.losses(model, clean, condition, times, masks).mean()

    @torch.no_grad()
    def sample(self, model, condition, shape, *, sampling_steps, generator,
               length_generator, temperature=1.):
        if len(condition) != 1 or temperature != 1.:
            raise ValueError("use sample_independent for batched D1; registered temperature is one")
        return sample_independent(model, self, condition, shape, sampling_steps,
                                  [generator], length_generators=[length_generator])


def profile_ids(protocol, pipeline):
    family = protocol["pipelines"][pipeline]["model"]
    return protocol["profile_selection"][family + "_order"]


def build_profile(protocol, pipeline, profile_id, shape, codec, device):
    if profile_id not in profile_ids(protocol, pipeline):
        raise ValueError(f"Unregistered profile {profile_id!r} for {pipeline}")
    spec = protocol["model_profiles"][profile_id]
    if spec["family"] == "gaussian":
        model = (CoordinateImageUNet(shape, spec["width"]) if spec["coordinates"]
                 else ImageUNet(2, 12, spec["width"]))
        diffusion = GaussianDiffusion(spec["diffusion_steps"], beta_start=spec["beta_start"],
                                      beta_end=spec["beta_end"], prediction_type=spec["prediction"], device=device)
    else:
        model = (LengthSequenceDenoiser(codec.vocabulary_size, width=spec["width"], embedding_dim=spec["embedding_dim"])
                 if profile_id == "D1" else SequenceDenoiser(codec.vocabulary_size, 32, 12,
                                                             width=spec["width"], embedding_dim=spec["embedding_dim"]))
        diffusion_class = LengthMaskedDiffusion if profile_id == "D1" else MaskedDiffusion
        diffusion = diffusion_class(codec.mask, torch.linspace(0, 1, spec["sampling_steps"] + 1), device=device)
    return model, diffusion


@torch.no_grad()
def sample_independent(model, diffusion, condition, shape, sampling_steps, generators, *, length_generators=None):
    """One RNG per candidate; length and payload have separate RNG streams."""
    if not len(condition) or len(condition) != len(generators) or not 1 <= sampling_steps <= diffusion.steps:
        raise ValueError("invalid trajectory count or sampling steps")
    device = diffusion.device
    if isinstance(diffusion, GaussianDiffusion):
        times = torch.linspace(diffusion.steps - 1, 0, sampling_steps, device=device).round().long().unique_consecutive()
        value = torch.stack([torch.randn(shape, device=device, generator=g) for g in generators])
        for position, index in enumerate(times):
            alpha = diffusion.alpha_bar[index]
            clock = torch.full((len(condition),), index.item() / (diffusion.steps - 1), device=device)
            output = model(value, clock, condition)
            clean = diffusion.predicted_clean(output, value, index.repeat(len(condition)))
            if diffusion.prediction_type == "sample":
                clean = clean.clamp(-1, 1)
                noise = (value - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            else:
                noise = output
            previous = diffusion.alpha_bar[times[position + 1]] if position + 1 < len(times) else torch.tensor(1., device=device)
            value = previous.sqrt() * clean + (1 - previous).sqrt() * noise
        value = value.clamp(-1, 1)
    else:
        value = torch.full((len(condition), *shape), diffusion.mask_token, dtype=torch.long, device=device)
        factorized = isinstance(diffusion, LengthMaskedDiffusion)
        if factorized:
            if shape != (32,) or length_generators is None or len(length_generators) != len(condition):
                raise ValueError("D1 requires a separate length RNG for every candidate")
            probabilities = model.length_head(condition).softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("non-finite length probabilities")
            lengths = torch.stack([torch.multinomial(row, 1, generator=g)[0]
                                   for row, g in zip(probabilities, length_generators)]) + 4
            positions = torch.arange(32, device=device)[None]
            value = torch.where(positions == lengths[:, None], diffusion.mask_token - 2, value)
            value = torch.where(positions > lengths[:, None], diffusion.mask_token - 1, value)
            condition = diffusion.payload_condition(condition, lengths)
        times = torch.linspace(diffusion.steps, 0, sampling_steps + 1, device=device).round().long()
        for current, previous in zip(times[:-1], times[1:]):
            clock = diffusion.probabilities[current].expand(len(condition))
            logits = model(value, clock, condition)
            if factorized:
                logits = logits[..., :diffusion.mask_token - 2]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite payload logits")
            tokens = torch.stack([torch.multinomial(row.softmax(-1), 1, generator=g).squeeze(-1)
                                  for row, g in zip(logits, generators)])
            probability = 1 - diffusion.probabilities[previous] / diffusion.probabilities[current]
            reveal = torch.stack([torch.rand(shape, device=device, generator=g) for g in generators]) < probability
            value = torch.where((value == diffusion.mask_token) & reveal, tokens, value)
    if not torch.isfinite(value).all():
        raise FloatingPointError("non-finite profile sampler")
    return value
