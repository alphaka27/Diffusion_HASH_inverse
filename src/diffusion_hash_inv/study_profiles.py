"""Registered v3.1 development profiles, built from existing primitives."""
from math import prod

import torch
from torch import nn

from .discrete import MaskedDiffusion, SequenceDenoiser
from .models import GaussianDiffusion, ImageUNet


class CoordinateImageUNet(ImageUNet):
    """Append public, fixed coordinates only to the denoiser's input."""

    def __init__(self, shape, width, *, condition_output=False, factorized_length=False):
        super().__init__(2, 13 if factorized_length else 12, width)
        self.condition_output = nn.Linear(width * 3, prod(shape)) if condition_output else None
        if factorized_length:
            self.length_head = nn.Linear(12, 28)
        self.input = nn.Conv2d(4, width, 3, padding=1)
        height, width_pixels = shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, height),
                              torch.linspace(-1, 1, width_pixels), indexing="ij")
        self.register_buffer("coordinates", torch.stack((x, y))[None], persistent=False)

    def forward(self, value, time, condition):
        if value.shape[1:] != (2, *self.coordinates.shape[-2:]):
            raise ValueError("coordinate profile expects the original two-channel image")
        output = super().forward(torch.cat((value, self.coordinates.expand(len(value), -1, -1, -1)), 1), time, condition)
        if self.condition_output is not None:
            embedding = self.condition(torch.cat((condition, time[:, None]), 1))
            output = output + self.condition_output(embedding).reshape_as(output)
        return output


class LengthSequenceDenoiser(SequenceDenoiser):
    def __init__(self, vocabulary_size, *, width, embedding_dim, condition_output=False):
        super().__init__(vocabulary_size, 32, 13, width=width, embedding_dim=embedding_dim)
        self.condition_output = nn.Linear(13, 32 * self.clean_states) if condition_output else None
        self.length_head = nn.Linear(12, 28)

    def forward(self, value, time, condition):
        output = super().forward(value, time, condition)
        return output if self.condition_output is None else output + self.condition_output(condition).reshape_as(output)


class RegionGaussianDiffusion(GaussianDiffusion):
    """Development-only x0 objective; preserve the frozen v3 base model."""

    def __init__(self, *args, loss_regions, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_regions not in {"bgv", "cgge"} or self.prediction_type != "sample":
            raise ValueError("region loss requires BGV/CGGE clean-sample prediction")
        self.loss_regions = loss_regions

    def losses(self, model, clean, condition, index, noise):
        width = 16 if self.loss_regions == "bgv" else 8
        if clean.shape[1:] != (2, 32, width * 8):
            raise ValueError("region loss requires the registered image shape")
        output = model(self.add_noise(clean, noise, index), index.float() / (self.steps - 1), condition)
        error = (output - clean).square()
        slots = torch.arange(4, device=clean.device)[:, None].repeat_interleave(8, 0) * 8
        slots = slots + torch.arange(8, device=clean.device)[None].repeat_interleave(width, 1)
        first = int(self.loss_regions == "bgv")
        active = clean[:, 1] > 0
        regions = [active & (slots >= first) & (slots < first + 3),
                   active & (slots >= first + 3), ~active]
        if first:
            regions.append((slots == 0).expand_as(active))
        # Equal region weights prevent suffix/background area from hiding errors.
        values = error[:, 1].mean((1, 2))
        for region in regions:
            values = values + (error[:, 0] * region).sum((1, 2)) / region.sum((1, 2)).clamp_min(1)
        return values

    def loss(self, model, clean, condition, *, generator):
        index = torch.randint(self.steps, (len(clean),), device=self.device, generator=generator)
        noise = torch.randn(clean.shape, device=self.device, generator=generator)
        return self.losses(model, clean, condition, index, noise).mean()


class LengthMaskedDiffusion(MaskedDiffusion):
    """Learn length, then diffuse payload with intrinsic EOS/PAD context."""

    factorized = True

    def __init__(self, *args, prefix_balanced_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_balanced_loss = prefix_balanced_loss

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
        if condition.shape != (len(lengths), 12):
            raise ValueError("length-factorized external condition contains only twelve public bits")
        return torch.cat((condition, lengths[:, None].to(condition.dtype) / 31), 1)

    def losses(self, model, clean, condition, times, masks, *, return_components=False):
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
        length_ce = nn.functional.cross_entropy(length_logits, lengths - 4, reduction="none")
        payload_ce = (payload * masks).sum(1) / masks.sum(1).clamp_min(1)
        components = {"length_ce": length_ce.mean()}
        if self.prefix_balanced_loss:
            prefix = masks & (torch.arange(32, device=clean.device)[None] < 3)
            suffix = masks & ~prefix
            prefix_ce = (payload * prefix).sum(1) / prefix.sum(1).clamp_min(1)
            suffix_ce = (payload * suffix).sum(1) / suffix.sum(1).clamp_min(1)
            payload_ce = prefix_ce + suffix_ce
            components.update(prefix_ce=prefix_ce.mean(), suffix_ce=suffix_ce.mean())
        components["payload_ce"] = payload_ce.mean()
        total = length_ce + payload_ce
        return (total, components) if return_components else total

    def loss(self, model, clean, condition, *, generator, return_components=False):
        times = torch.rand((len(clean),), device=self.device, generator=generator)
        masks = torch.rand(clean.shape, device=self.device, generator=generator) < times[:, None]
        values = self.losses(model, clean, condition, times, masks, return_components=return_components)
        return (values[0].mean(), values[1]) if return_components else values.mean()

    @torch.no_grad()
    def sample(self, model, condition, shape, *, sampling_steps, generator,
               length_generator, temperature=1.):
        if len(condition) != 1 or temperature != 1.:
            raise ValueError("use sample_independent for batched D1; registered temperature is one")
        return sample_independent(model, self, condition, shape, sampling_steps,
                                  [generator], length_generators=[length_generator])


class LengthGaussianDiffusion(RegionGaussianDiffusion):
    """Diffuse glyph payload inside a single learned-length structural context."""

    factorized = True
    payload_condition = staticmethod(LengthMaskedDiffusion.payload_condition)

    def structure(self, lengths):
        width = 16 if self.loss_regions == "bgv" else 8
        first = int(self.loss_regions == "bgv")
        y = torch.arange(32, device=self.device)[:, None]
        x = torch.arange(width * 8, device=self.device)[None]
        slots = y // 8 * 8 + x // width
        active = slots < lengths[:, None, None] + first
        payload = active & (slots >= first)
        glyph = torch.full(active.shape, -1., device=self.device)
        if first:
            shifts = 7 - (y % 8 // 4 * 4 + x % 16 // 4)
            header = ((lengths[:, None, None] >> shifts) & 1).float() * 2 - 1
            glyph = torch.where(slots == 0, header, glyph)
        fixed = torch.stack((glyph, active.float() * 2 - 1), 1)
        return fixed, torch.stack((payload, torch.zeros_like(payload)), 1)

    def lengths(self, clean):
        width = 16 if self.loss_regions == "bgv" else 8
        if clean.shape[1:] != (2, 32, width * 8) or not torch.isfinite(clean).all():
            raise ValueError("G3 requires finite encoded images")
        lengths = (clean[:, 1] > 0).sum((1, 2)) // (8 * width) - int(self.loss_regions == "bgv")
        fixed, payload = self.structure(lengths)
        if ((lengths < 4) | (lengths > 31)).any() or not torch.equal(clean[~payload], fixed[~payload]):
            raise ValueError("G3 training image has invalid length/header/mask/padding")
        return lengths

    def add_noise(self, clean, noise, index):
        fixed, payload = self.structure(self.lengths(clean))
        return torch.where(payload, super().add_noise(clean, noise, index), fixed)

    def losses(self, model, clean, condition, index, noise, *, return_components=False):
        lengths = self.lengths(clean)
        output = model(self.add_noise(clean, noise, index), index.float() / (self.steps - 1),
                       self.payload_condition(condition, lengths))
        length_logits = model.length_head(condition)
        if not torch.isfinite(output).all() or not torch.isfinite(length_logits).all():
            raise FloatingPointError("non-finite G3 prediction")
        width = 16 if self.loss_regions == "bgv" else 8
        slots = torch.arange(32, device=self.device)[:, None] // 8 * 8 + torch.arange(width * 8, device=self.device)[None] // width
        first = int(self.loss_regions == "bgv")
        payload = self.structure(lengths)[1][:, 0]
        error = (output[:, 0] - clean[:, 0]).square()
        prefix = payload & (slots < first + 3)
        suffix = payload & ~prefix
        prefix_mse = (error * prefix).sum((1, 2)) / prefix.sum((1, 2)).clamp_min(1)
        suffix_mse = (error * suffix).sum((1, 2)) / suffix.sum((1, 2)).clamp_min(1)
        length_ce = nn.functional.cross_entropy(length_logits, lengths - 4, reduction="none")
        total = length_ce + prefix_mse + suffix_mse
        parts = {"length_ce": length_ce.mean(), "prefix_mse": prefix_mse.mean(), "suffix_mse": suffix_mse.mean()}
        return (total, parts) if return_components else total

    def loss(self, model, clean, condition, *, generator, return_components=False):
        index = torch.randint(self.steps, (len(clean),), device=self.device, generator=generator)
        noise = torch.randn(clean.shape, device=self.device, generator=generator)
        values = self.losses(model, clean, condition, index, noise, return_components=return_components)
        return (values[0].mean(), values[1]) if return_components else values.mean()

    @torch.no_grad()
    def sample(self, model, condition, shape, *, sampling_steps, generator, length_generator):
        if len(condition) != 1:
            raise ValueError("use sample_independent for batched G3")
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
        factorized = spec.get("factorized_length", False)
        model = (CoordinateImageUNet(shape, spec["width"], condition_output=spec.get("condition_output", False),
                                    factorized_length=factorized) if spec["coordinates"]
                 else ImageUNet(2, 12, spec["width"]))
        weighted = spec.get("region_balanced_loss", False)
        diffusion_class = LengthGaussianDiffusion if factorized else RegionGaussianDiffusion if weighted else GaussianDiffusion
        options = {"loss_regions": protocol["pipelines"][pipeline]["representation"]} if weighted or factorized else {}
        diffusion = diffusion_class(spec["diffusion_steps"], beta_start=spec["beta_start"],
                                    beta_end=spec["beta_end"], prediction_type=spec["prediction"], device=device, **options)
    else:
        model = (LengthSequenceDenoiser(codec.vocabulary_size, width=spec["width"], embedding_dim=spec["embedding_dim"],
                                       condition_output=spec.get("condition_output", False))
                 if profile_id == "D1" else SequenceDenoiser(codec.vocabulary_size, 32, 12,
                                                             width=spec["width"], embedding_dim=spec["embedding_dim"]))
        diffusion_class = LengthMaskedDiffusion if profile_id == "D1" else MaskedDiffusion
        options = {"prefix_balanced_loss": spec.get("prefix_balanced_loss", False)} if profile_id == "D1" else {}
        diffusion = diffusion_class(codec.mask, torch.linspace(0, 1, spec["sampling_steps"] + 1), device=device, **options)
    return model, diffusion


@torch.no_grad()
def sample_independent(model, diffusion, condition, shape, sampling_steps, generators, *, length_generators=None, trace=None):
    """One RNG per candidate; length and payload have separate RNG streams."""
    if not len(condition) or len(condition) != len(generators) or not 1 <= sampling_steps <= diffusion.steps:
        raise ValueError("invalid trajectory count or sampling steps")
    device = diffusion.device
    if isinstance(diffusion, GaussianDiffusion):
        factorized = isinstance(diffusion, LengthGaussianDiffusion)
        if factorized:
            if length_generators is None or len(length_generators) != len(condition):
                raise ValueError("G3 requires a separate length RNG for every candidate")
            probabilities = model.length_head(condition).softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("non-finite length probabilities")
            lengths = torch.stack([torch.multinomial(row, 1, generator=g)[0]
                                   for row, g in zip(probabilities, length_generators)]) + 4
            fixed, payload = diffusion.structure(lengths)
            if tuple(fixed.shape[1:]) != tuple(shape):
                raise ValueError("G3 image shape does not match representation")
            condition = diffusion.payload_condition(condition, lengths)
        times = torch.linspace(diffusion.steps - 1, 0, sampling_steps, device=device).round().long().unique_consecutive()
        value = torch.stack([torch.randn(shape, device=device, generator=g) for g in generators])
        if factorized:
            value = torch.where(payload, value, fixed)
        for position, index in enumerate(times):
            alpha = diffusion.alpha_bar[index]
            clock = torch.full((len(condition),), index.item() / (diffusion.steps - 1), device=device)
            output = model(value, clock, condition)
            if not torch.isfinite(output).all():
                raise FloatingPointError("non-finite Gaussian prediction")
            clean = diffusion.predicted_clean(output, value, index.repeat(len(condition)))
            if diffusion.prediction_type == "sample":
                clean = clean.clamp(-1, 1)
                noise = (value - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            else:
                noise = output
            previous = diffusion.alpha_bar[times[position + 1]] if position + 1 < len(times) else torch.tensor(1., device=device)
            value = previous.sqrt() * clean + (1 - previous).sqrt() * noise
            if factorized:
                value = torch.where(payload, value, fixed)
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
        for step, (current, previous) in enumerate(zip(times[:-1], times[1:]), 1):
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
            reveal = (value == diffusion.mask_token) & reveal
            if trace is not None:
                trace(step, float(clock[0]), logits[:, :3], tokens[:, :3], reveal[:, :3])
            value = torch.where(reveal, tokens, value)
    if not torch.isfinite(value).all():
        raise FloatingPointError("non-finite profile sampler")
    return value
