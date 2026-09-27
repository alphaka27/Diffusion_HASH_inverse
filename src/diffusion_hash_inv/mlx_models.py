"""Native MLX v3.1 denoisers and diffusion; no PyTorch model operations.

Images enter/leave as NCHW for the existing codecs; convolutions use NHWC.
Every corruption and sampling trajectory uses an explicit 64-bit PRNG key.
"""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np


def condition(values):
    values = list(values)
    if not values or any(type(y) is not int or not 0 <= y < 4096 for y in values):
        raise ValueError("Only public 12-bit conditions may enter the model")
    return mx.array([[(y >> shift) & 1 for shift in range(11, -1, -1)] for y in values], dtype=mx.float32)


def finite(values):
    leaves = [value for _, value in tree_flatten(values)]
    return all(bool(mx.all(mx.isfinite(value)).item()) for value in leaves)


def _initialize(layer, fan_in):
    """Match the reference initialization distributions, not its RNG bitstream."""
    bound = 1 / math.sqrt(fan_in)
    layer.weight = mx.random.uniform(-bound, bound, shape=layer.weight.shape)
    layer.bias = mx.random.uniform(-bound, bound, shape=layer.bias.shape)
    return layer


class ImageUNet(nn.Module):
    def __init__(self, shape, width=32, *, coordinates=False):
        super().__init__()
        if len(shape) != 3 or shape[0] != 2 or any(side % 2 for side in shape[1:]):
            raise ValueError("two-channel images with even spatial dimensions required")
        self._shape = tuple(shape)
        self._coordinates = coordinates
        channels = 4 if coordinates else 2
        self.input = _initialize(nn.Conv2d(channels, width, 3, padding=1), channels * 9)
        self.down = _initialize(nn.Conv2d(width, width * 2, 4, stride=2, padding=1), width * 16)
        self.middle = _initialize(nn.Conv2d(width * 2, width * 2, 3, padding=1), width * 2 * 9)
        self.up = _initialize(nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1), width * 16)
        self.output = _initialize(nn.Conv2d(width * 2, 2, 3, padding=1), width * 2 * 9)
        self.condition = nn.Sequential(_initialize(nn.Linear(13, width * 3), 13), nn.SiLU(),
                                       _initialize(nn.Linear(width * 3, width * 3), width * 3))

    def __call__(self, value, time, condition):
        if value.shape[1:] != self._shape or time.shape != (len(value),) or condition.shape != (len(value), 12):
            raise ValueError("invalid image/time/public condition shape")
        value = value.transpose(0, 2, 3, 1)
        if self._coordinates:
            y, x = mx.meshgrid(mx.linspace(-1, 1, value.shape[1]), mx.linspace(-1, 1, value.shape[2]), indexing="ij")
            grid = mx.broadcast_to(mx.stack((x, y), -1)[None], (*value.shape[:3], 2))
            value = mx.concatenate((value, grid), -1)
        embedding = self.condition(mx.concatenate((condition, time[:, None]), -1))
        width = self.input.weight.shape[0]
        skip = nn.silu(self.input(value) + embedding[:, None, None, :width])
        hidden = nn.silu(self.down(skip) + embedding[:, None, None, width:])
        hidden = nn.silu(self.middle(hidden))
        hidden = nn.silu(self.up(hidden))
        return self.output(mx.concatenate((hidden, skip), -1)).transpose(0, 3, 1, 2)


class DenseEmbedding(nn.Embedding):
    def __call__(self, tokens):
        # Same-backend recovery avoids repeated-token atomic scatter gradients.
        one_hot = (tokens[..., None] == mx.arange(self.weight.shape[0])).astype(self.weight.dtype)
        return one_hot @ self.weight


class SequenceDenoiser(nn.Module):
    def __init__(self, vocabulary_size, *, width=128, embedding_dim=16, factorized=False):
        super().__init__()
        self._condition_dim = 13 if factorized else 12
        self._clean_states = vocabulary_size - 1
        self.embedding = DenseEmbedding(vocabulary_size, embedding_dim)
        self.embedding.weight = mx.random.normal(self.embedding.weight.shape)
        inputs = 32 * embedding_dim + self._condition_dim + 1
        self.network = nn.Sequential(_initialize(nn.Linear(inputs, width), inputs), nn.SiLU(),
                                     _initialize(nn.Linear(width, 32 * self._clean_states), width))
        if factorized:
            self.length_head = _initialize(nn.Linear(12, 28), 12)

    def __call__(self, value, time, condition):
        if value.shape != (len(value), 32) or condition.shape != (len(value), self._condition_dim) or time.shape != (len(value),):
            raise ValueError("invalid token/time/condition shape")
        inputs = mx.concatenate((self.embedding(value).reshape(len(value), -1), time[:, None], condition), -1)
        return self.network(inputs).reshape(len(value), 32, self._clean_states)


class GaussianDiffusion:
    backend = "mlx"
    discrete = False

    def __init__(self, steps, *, beta_start=1e-4, beta_end=.02, prediction_type="epsilon"):
        if steps < 2 or not 0 < beta_start < beta_end < 1 or prediction_type not in {"epsilon", "sample"}:
            raise ValueError("invalid Gaussian diffusion settings")
        self.steps, self.prediction_type = steps, prediction_type
        self.alpha_bar = mx.cumprod(1 - mx.linspace(beta_start, beta_end, steps, dtype=mx.float32))

    def add_noise(self, clean, noise, indices):
        alpha = self.alpha_bar[indices].reshape((-1, 1, 1, 1))
        return mx.sqrt(alpha) * clean + mx.sqrt(1 - alpha) * noise

    def predicted_clean(self, output, noisy, indices):
        if self.prediction_type == "sample":
            return output
        alpha = self.alpha_bar[indices].reshape((-1, 1, 1, 1))
        return (noisy - mx.sqrt(1 - alpha) * output) / mx.sqrt(alpha)

    def noise_inputs(self, clean, keys):
        split = [mx.random.split(key) for key in keys]
        indices = mx.stack([mx.random.randint(0, self.steps, key=k[0]) for k in split])
        noise = mx.stack([mx.random.normal(clean.shape[1:], key=k[1]) for k in split])
        return indices, noise

    def losses(self, model, clean, condition, indices, noise):
        output = model(self.add_noise(clean, noise, indices), indices.astype(mx.float32) / (self.steps - 1), condition)
        target = noise if self.prediction_type == "epsilon" else clean
        return mx.mean(mx.square(output - target), axis=(1, 2, 3))

    def loss(self, model, clean, condition, *, key):
        indices, noise = self.noise_inputs(clean, mx.random.split(key, len(clean)))
        return self.losses(model, clean, condition, indices, noise).mean()


class MaskedDiffusion:
    backend = "mlx"
    discrete = True

    def __init__(self, mask_token, steps, *, factorized=False):
        if mask_token < 3 or steps < 1:
            raise ValueError("invalid masked diffusion settings")
        self.mask_token, self.steps, self.factorized = mask_token, steps, factorized
        self.probabilities = mx.linspace(0, 1, steps + 1, dtype=mx.float32)

    def lengths(self, clean):
        return mx.argmax(clean == self.mask_token - 2, axis=1)

    def validate_clean(self, clean):
        if clean.ndim != 2 or clean.shape[1] != 32 or not mx.issubdtype(clean.dtype, mx.integer):
            raise ValueError("32-position integer sequences required")
        if mx.any((clean < 0) | (clean >= self.mask_token)).item():
            raise ValueError("clean tokens exclude MASK and unknown states")
        if self.factorized:
            lengths = self.lengths(clean)
            positions = mx.arange(32)[None]
            invalid = ((clean == self.mask_token - 2).sum(1) != 1) | (lengths < 4) | (lengths > 31)
            invalid = invalid | mx.any((positions < lengths[:, None]) & (clean >= self.mask_token - 2), axis=1)
            invalid = invalid | mx.any((positions > lengths[:, None]) & (clean != self.mask_token - 1), axis=1)
            if mx.any(invalid).item():
                raise ValueError("invalid D1 length/EOS/payload/PAD training record")

    @staticmethod
    def payload_condition(condition, lengths):
        if condition.shape != (len(lengths), 12):
            raise ValueError("D1 external condition contains twelve public bits only")
        return mx.concatenate((condition, lengths[:, None].astype(mx.float32) / 31), -1)

    def noise_inputs(self, clean, keys):
        split = [mx.random.split(key) for key in keys]
        times = mx.stack([mx.random.uniform(key=k[0]) for k in split])
        masks = mx.stack([mx.random.uniform(shape=clean.shape[1:], key=k[1]) for k in split]) < times[:, None]
        return times, masks

    def losses(self, model, clean, condition, times, masks, *, return_components=False):
        lengths = self.lengths(clean) if self.factorized else None
        if self.factorized:
            masks = masks & (mx.arange(32)[None] < lengths[:, None])
        cond = self.payload_condition(condition, lengths) if self.factorized else condition
        logits = model(mx.where(masks, self.mask_token, clean), times, cond)
        targets = clean
        if self.factorized:
            logits = logits[..., :self.mask_token - 2]
            targets = mx.minimum(clean, self.mask_token - 3)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        values = (losses * masks).sum(1) / mx.maximum(masks.sum(1), 1)
        components = {"payload_ce": values.mean()}
        if self.factorized:
            length_ce = nn.losses.cross_entropy(model.length_head(condition), lengths - 4, reduction="none")
            components["length_ce"] = length_ce.mean()
            values = values + length_ce
        return (values, components) if return_components else values

    def loss(self, model, clean, condition, *, key, return_components=False):
        times, masks = self.noise_inputs(clean, mx.random.split(key, len(clean)))
        values = self.losses(model, clean, condition, times, masks, return_components=return_components)
        return (values[0].mean(), values[1]) if return_components else values.mean()


def build_profile(protocol, pipeline, profile_id, *, seed):
    cfg = protocol["pipelines"][pipeline]
    if profile_id not in protocol["profile_selection"][cfg["model"] + "_order"]:
        raise ValueError(f"Unregistered MLX profile {profile_id!r} for {pipeline}")
    spec = protocol["model_profiles"][profile_id]
    mx.random.seed(seed)
    if cfg["model"] == "gaussian":
        shape = (2, 32, 64 if cfg["representation"] == "cgge" else 128)
        model = ImageUNet(shape, spec["width"], coordinates=spec["coordinates"])
        diffusion = GaussianDiffusion(spec["diffusion_steps"], beta_start=spec["beta_start"],
                                      beta_end=spec["beta_end"], prediction_type=spec["prediction"])
    else:
        tokens = protocol["codecs"]["tokens"][cfg["source"]]
        model = SequenceDenoiser(tokens["vocabulary_size"], width=spec["width"],
                                 embedding_dim=spec["embedding_dim"], factorized=profile_id == "D1")
        diffusion = MaskedDiffusion(tokens["mask"], spec["sampling_steps"], factorized=profile_id == "D1")
    mx.eval(model.parameters())
    return model, diffusion


def sample(model, diffusion, cond, shape, *, steps, seeds, length_seeds=None):
    if not len(cond) or len(cond) != len(seeds) or not 1 <= steps <= diffusion.steps:
        raise ValueError("invalid sampling steps or trajectory count")
    model.eval()
    keys = [mx.random.key(seed) for seed in seeds]
    if not diffusion.discrete:
        times = np.rint(np.linspace(diffusion.steps - 1, 0, steps, dtype=np.float32)).astype(int).tolist()
        value = mx.stack([mx.random.normal(shape, key=key) for key in keys])
        for position, index in enumerate(times):
            output = model(value, mx.full((len(cond),), index / (diffusion.steps - 1)), cond)
            if not finite(output):
                raise FloatingPointError("non-finite Gaussian prediction")
            clean = diffusion.predicted_clean(output, value, mx.full((len(cond),), index, dtype=mx.int32))
            if diffusion.prediction_type == "sample":
                clean = mx.clip(clean, -1, 1)
                alpha = diffusion.alpha_bar[index]
                noise = (value - mx.sqrt(alpha) * clean) / mx.sqrt(1 - alpha)
            else:
                noise = output
            previous = diffusion.alpha_bar[times[position + 1]] if position + 1 < len(times) else mx.array(1.)
            value = mx.sqrt(previous) * clean + mx.sqrt(1 - previous) * noise
            mx.eval(value)
        if not finite(value):
            raise FloatingPointError("non-finite Gaussian trajectory")
        value = mx.clip(value, -1, 1)
    else:
        value = mx.full((len(cond), *shape), diffusion.mask_token, dtype=mx.int32)
        if diffusion.factorized:
            if shape != (32,) or length_seeds is None or len(length_seeds) != len(cond):
                raise ValueError("D1 requires a separate length seed for each trajectory")
            logits = model.length_head(cond)
            if not finite(logits):
                raise FloatingPointError("non-finite length logits")
            lengths = mx.stack([mx.random.categorical(row, key=mx.random.key(seed))
                                for row, seed in zip(logits, length_seeds)]) + 4
            positions = mx.arange(32)[None]
            value = mx.where(positions == lengths[:, None], diffusion.mask_token - 2, value)
            value = mx.where(positions > lengths[:, None], diffusion.mask_token - 1, value)
            cond = diffusion.payload_condition(cond, lengths)
        keys = [mx.random.split(key, steps * 2) for key in keys]
        times = np.rint(np.linspace(diffusion.steps, 0, steps + 1)).astype(int).tolist()
        for position, (current, previous) in enumerate(zip(times[:-1], times[1:])):
            logits = model(value, mx.broadcast_to(diffusion.probabilities[current], (len(cond),)), cond)
            if diffusion.factorized:
                logits = logits[..., :diffusion.mask_token - 2]
            if not finite(logits):
                raise FloatingPointError("non-finite categorical logits")
            tokens = mx.stack([mx.random.categorical(row, key=key[2 * position]) for row, key in zip(logits, keys)])
            probability = 1 - diffusion.probabilities[previous] / diffusion.probabilities[current]
            reveal = mx.stack([mx.random.uniform(shape=shape, key=key[2 * position + 1]) for key in keys]) < probability
            value = mx.where((value == diffusion.mask_token) & reveal, tokens, value)
            mx.eval(value)
    mx.eval(value)
    return value
