"""독립 MLX 모델. D1은 dhi_v5/models.py, G3는 v3.1 mlx_models.py에서 복사했다."""
import math

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .data import SOURCES, TOKENS, WIDTH, condition_bits, key_words, seed
from .protocol import registration
from . import codecs, data

ALPHA_BAR = mx.cumprod(1 - mx.linspace(1e-4, .02, 1000, dtype=mx.float32))



class Denoiser(nn.Module):
    def __init__(self, architecture="D1-S", src="P"):
        super().__init__()
        self.architecture = architecture
        self.src = src
        self.representation = "tokens"
        self.states = SOURCES[src]["states"]
        self.vocab = self.states + 3
        self.length_head = nn.Linear(12, 28)
        if architecture == "D1-S":
            self.embedding = nn.Embedding(self.vocab, 16)
            self.hidden = nn.Linear(WIDTH * 16 + 14, 128)
            self.output = nn.Linear(128, WIDTH * (self.states + 2))
            self.residual = nn.Linear(13, WIDTH * (self.states + 2))
        elif architecture == "D1-T-L" and src == "P":
            dim, layers, heads = 256, 8, 8
            self.embedding = nn.Embedding(self.vocab, dim)
            self.position = nn.Embedding(WIDTH, dim)
            self.context = nn.Linear(14, dim)
            self.blocks = [nn.TransformerEncoderLayer(dim, heads, 4 * dim, activation=nn.gelu, norm_first=True) for _ in range(layers)]
            self.norm = nn.LayerNorm(dim)
            self.output = nn.Linear(dim, self.states + 2)
            self.residual = nn.Linear(13, WIDTH * (self.states + 2))
        else:
            raise ValueError("Architecture must be D1-S, D1-T or D1-T-L")

    def __call__(self, tokens, time, condition, lengths):
        c = mx.concatenate((condition, lengths[:, None].astype(mx.float32) / 31), axis=1)
        context = mx.concatenate((c, time[:, None]), axis=1)
        # Dense one-hot multiplication avoids nondeterministic Metal scatter-add
        # in embedding gradients, so interrupted training reproduces bitwise.
        h = (tokens[:, :, None] == mx.arange(self.vocab)).astype(mx.float32) @ self.embedding.weight
        if self.architecture == "D1-S":
            h = nn.silu(self.hidden(mx.concatenate((h.reshape(len(tokens), -1), context), axis=1)))
            logits = self.output(h).reshape(-1, WIDTH, self.states + 2)
        else:
            h = h + self.position(mx.arange(WIDTH))[None, :, :] + self.context(context)[:, None, :]
            padding = mx.arange(WIDTH)[None, :] > lengths[:, None]
            attention_mask = mx.where(padding[:, None, None, :], -1e9, 0.0)
            for block in self.blocks:
                h = block(h, attention_mask)
            logits = self.output(self.norm(h))
        # EOS/PAD have model logits but are placed only by the sampled length.
        return (logits + self.residual(c).reshape(-1, WIDTH, self.states + 2))[:, :, :self.states]


def _initialize(layer, fan_in):
    """Match the reference initialization distributions, not its RNG bitstream."""
    bound = 1 / math.sqrt(fan_in)
    layer.weight = mx.random.uniform(-bound, bound, shape=layer.weight.shape)
    layer.bias = mx.random.uniform(-bound, bound, shape=layer.bias.shape)
    return layer


class ImageUNet(nn.Module):
    def __init__(self, shape, width=32, *, coordinates=False, condition_output=False, factorized_length=False):
        super().__init__()
        if len(shape) != 3 or shape[0] != 2 or any(side % 2 for side in shape[1:]):
            raise ValueError("two-channel images with even spatial dimensions required")
        self._shape = tuple(shape)
        self._coordinates = coordinates
        self._condition_dim = 13 if factorized_length else 12
        channels = 4 if coordinates else 2
        self.input = _initialize(nn.Conv2d(channels, width, 3, padding=1), channels * 9)
        self.down = _initialize(nn.Conv2d(width, width * 2, 4, stride=2, padding=1), width * 16)
        self.middle = _initialize(nn.Conv2d(width * 2, width * 2, 3, padding=1), width * 2 * 9)
        self.up = _initialize(nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1), width * 16)
        self.output = _initialize(nn.Conv2d(width * 2, 2, 3, padding=1), width * 2 * 9)
        self.condition = nn.Sequential(_initialize(nn.Linear(self._condition_dim + 1, width * 3), self._condition_dim + 1), nn.SiLU(),
                                       _initialize(nn.Linear(width * 3, width * 3), width * 3))
        if condition_output:
            self.condition_output = _initialize(nn.Linear(width * 3, math.prod(shape)), width * 3)
        if factorized_length:
            self.length_head = _initialize(nn.Linear(12, 28), 12)

    def __call__(self, value, time, condition):
        if value.shape[1:] != self._shape or time.shape != (len(value),) or condition.shape != (len(value), self._condition_dim):
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
        output = self.output(mx.concatenate((hidden, skip), -1)).transpose(0, 3, 1, 2)
        return output + self.condition_output(embedding).reshape(output.shape) if hasattr(self, "condition_output") else output


def make_model(pipeline, stage, seed_id, architecture=None):
    spec = registration()["pipelines"][pipeline]
    architecture = architecture or spec["model"]
    mx.random.seed(seed("weights", stage, pipeline, seed_id) & 0xFFFFFFFF)
    if architecture == "G3-U":
        model = ImageUNet(codecs.SHAPES[spec["representation"]], 32, coordinates=True,
                          condition_output=True, factorized_length=True)
        model.architecture, model.src, model.representation = architecture, spec["source"], spec["representation"]
    else:
        model = Denoiser(architecture, spec["source"])
    mx.eval(model.parameters())
    return model


def parameter_count(model):
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def candidate_keys(namespace, indices):
    """SplitMix64 maps stable row ordinals to explicit two-word MLX keys."""
    return mx.array(key_words(namespace, indices))


def fork_keys(keys, draw):
    return draw_key(keys, draw)


def draw_key(key, draw):
    # Explicit draw namespaces also avoid nested-vmap split/indexing differences.
    return key ^ mx.array([(0x9E3779B9 * (draw + 1)) & 0xFFFFFFFF,
                           (0x85EBCA6B * (draw + 1)) & 0xFFFFFFFF], dtype=mx.uint32)


def corrupt(tokens, lengths, keys, draw=0, src="P"):
    time = mx.vmap(lambda k: mx.random.uniform(key=k))(fork_keys(keys, 2 * draw))
    uniforms = mx.vmap(lambda k: mx.random.uniform(shape=(WIDTH,), key=k))(fork_keys(keys, 2 * draw + 1))
    masked = (uniforms < time[:, None]) & (mx.arange(WIDTH)[None, :] < lengths[:, None])
    return mx.where(masked, TOKENS[src]["mask"], tokens), time, masked


def loss_rows(model, tokens, lengths, conditions, corruption):
    noisy, time, masked = corruption
    logits = model(noisy, time, conditions, lengths)
    ce = nn.losses.cross_entropy(logits, mx.clip(tokens, 0, model.states - 1), reduction="none")
    payload = mx.sum(ce * masked, axis=1) / mx.maximum(mx.sum(masked, axis=1), 1)
    return nn.losses.cross_entropy(model.length_head(conditions), lengths - 4, reduction="none") + payload


def sample_tokens(model, labels, keys):
    """32 reverse intervals, one sampled length, no remasking; NFE=33."""
    PAD, EOS, MASK = (TOKENS[model.src][k] for k in ("pad", "eos", "mask"))
    conditions = mx.array(condition_bits(labels))
    if keys.shape != (len(labels), 2):
        raise ValueError("One explicit RNG key per candidate is required")
    lengths = mx.vmap(lambda logits, key: mx.random.categorical(logits, key=key))(
        model.length_head(conditions), fork_keys(keys, 0)).astype(mx.int32) + 4
    positions = mx.arange(WIDTH)[None, :]
    tokens = mx.where(positions < lengths[:, None], MASK, mx.where(positions == lengths[:, None], EOS, PAD))
    for remaining in range(32, 0, -1):
        logits = model(tokens, mx.full((len(labels),), remaining / 32), conditions, lengths)
        choices = mx.vmap(lambda row, key: mx.random.categorical(row, key=key))(
            logits, fork_keys(keys, 2 * remaining - 1))
        uniform = mx.vmap(lambda key: mx.random.uniform(shape=(WIDTH,), key=key))(fork_keys(keys, 2 * remaining))
        reveal = (tokens == MASK) & (uniform < 1 / remaining)
        tokens = mx.where(reveal, choices.astype(mx.int32), tokens)
        mx.eval(tokens)
    return np.asarray(tokens)


def sample_tokens_reference(model, labels, keys):
    """New scalar oracle for the V5 rewrite gate, never calls a legacy sampler."""
    PAD, EOS, MASK = (TOKENS[model.src][k] for k in ("pad", "eos", "mask"))
    rows = []
    for label, key in zip(labels, keys, strict=True):
        c = mx.array(condition_bits(np.array([label], dtype=np.int32)))
        length = int(mx.random.categorical(model.length_head(c)[0], key=draw_key(key, 0))) + 4
        tokens = mx.array([[MASK] * length + [EOS] + [PAD] * (31 - length)])
        for step in range(32, 0, -1):
            logits = model(tokens, mx.array([step / 32]), c, mx.array([length]))[0]
            choices = mx.random.categorical(logits, key=draw_key(key, 2 * step - 1))
            uniforms = mx.random.uniform(shape=(WIDTH,), key=draw_key(key, 2 * step))
            tokens = mx.where((tokens == MASK) & (uniforms[None, :] < 1 / step), choices[None, :].astype(mx.int32), tokens)
            mx.eval(tokens)
        rows.append(np.asarray(tokens[0]))
    return np.stack(rows)


def structure(lengths, representation):
    width = 16 if representation == "bgv" else 8
    first = int(representation == "bgv")
    y, x = mx.arange(32)[:, None], mx.arange(width * 8)[None]
    slots = y // 8 * 8 + x // width
    active = slots < lengths[:, None, None] + first
    payload = active & (slots >= first)
    glyph = mx.full(active.shape, -1.)
    if first:
        shifts = 7 - (y % 8 // 4 * 4 + x % 16 // 4)
        header = ((lengths[:, None, None] >> shifts) & 1).astype(mx.float32) * 2 - 1
        glyph = mx.where(slots == 0, header, glyph)
    return mx.stack((glyph, active.astype(mx.float32) * 2 - 1), 1), mx.stack((payload, mx.zeros_like(payload)), 1)


def gaussian_corruption(clean, lengths, keys, representation, draw=0):
    time = mx.vmap(lambda k: mx.random.randint(0, 1000, key=k))(draw_key(keys, 2 * draw))
    noise = mx.vmap(lambda k: mx.random.normal(clean.shape[1:], key=k))(draw_key(keys, 2 * draw + 1))
    fixed, payload = structure(lengths, representation)
    alpha = ALPHA_BAR[time][:, None, None, None]
    noisy = mx.where(payload, mx.sqrt(alpha) * clean + mx.sqrt(1 - alpha) * noise, fixed)
    return noisy, time.astype(mx.float32) / 999, payload


def gaussian_loss_rows(model, clean, lengths, conditions, corruption):
    noisy, time, payload = corruption
    condition = mx.concatenate((conditions, lengths[:, None].astype(mx.float32) / 31), axis=1)
    output = model(noisy, time, condition)
    error = mx.square(output[:, 0] - clean[:, 0]) * payload[:, 0]
    mse = error.sum((1, 2)) / mx.maximum(payload[:, 0].sum((1, 2)), 1)
    return nn.losses.cross_entropy(model.length_head(conditions), lengths - 4, reduction="none") + mse


def train_step(model, optimizer, clean, lengths, labels, keys, lr, update):
    clean, lengths = mx.array(clean), mx.array(lengths)
    conditions = mx.array(condition_bits(labels))
    if model.architecture == "G3-U":
        corruption = gaussian_corruption(clean, lengths, keys, model.representation)
        row_loss = gaussian_loss_rows
    else:
        corruption = corrupt(clean, lengths, keys, src=model.src)
        row_loss = loss_rows
    def objective(m):
        return row_loss(m, clean, lengths, conditions, corruption).mean()
    loss, grads = nn.value_and_grad(model, objective)(model)
    if model.architecture != "D1-S":
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.learning_rate = lr * min((update + 1) / 1000, 1)
    else:
        norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))
        optimizer.learning_rate = lr
    mx.eval(loss, norm)
    if not np.isfinite(float(loss)) or not np.isfinite(float(norm)):
        raise FloatingPointError("Non-finite loss/gradient; checkpoint not advanced")
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return float(loss)


def sample_images(model, labels, keys, steps=25):
    if len(labels) < 64 or len(labels) % 64:
        raise ValueError("Gaussian generation batch must be a positive multiple of 64")
    if steps not in (25, 50, 100) or keys.shape != (len(labels), 2):
        raise ValueError("Invalid sampler steps or candidate keys")
    conditions = mx.array(condition_bits(labels))
    lengths = mx.vmap(lambda logits, key: mx.random.categorical(logits, key=key))(
        model.length_head(conditions), draw_key(keys, 0)).astype(mx.int32) + 4
    fixed, payload = structure(lengths, model.representation)
    noise = mx.vmap(lambda k: mx.random.normal(model._shape, key=k))(draw_key(keys, 1))
    x = mx.where(payload, noise, fixed)
    condition = mx.concatenate((conditions, lengths[:, None].astype(mx.float32) / 31), axis=1)
    times = np.rint(np.linspace(999, 0, steps, dtype=np.float32)).astype(int)
    for position, time in enumerate(times):
        clean = mx.clip(model(x, mx.full((len(labels),), int(time) / 999), condition), -1, 1)
        alpha = ALPHA_BAR[int(time)]
        next_alpha = ALPHA_BAR[int(times[position + 1])] if position + 1 < steps else mx.array(1., dtype=mx.float32)
        epsilon = (x - mx.sqrt(alpha) * clean) / mx.sqrt(1 - alpha)
        x = mx.where(payload, mx.sqrt(next_alpha) * clean + mx.sqrt(1 - next_alpha) * epsilon, fixed)
        mx.eval(x)
        if not bool(mx.all(mx.isfinite(x))):
            raise FloatingPointError("Non-finite Gaussian sample")
    return np.asarray(mx.clip(x, -1, 1)), np.asarray(lengths)


def sample_images_reference(model, labels, keys, steps=25):
    images, lengths = [], []
    times = np.rint(np.linspace(999, 0, steps, dtype=np.float32)).astype(int)
    for label, key in zip(labels, keys, strict=True):
        c = mx.array(condition_bits(np.array([label], dtype=np.int32)))
        length = int(mx.random.categorical(model.length_head(c)[0], key=draw_key(key, 0))) + 4
        fixed, payload = structure(mx.array([length]), model.representation)
        x = mx.where(payload, mx.random.normal(model._shape, key=draw_key(key, 1))[None], fixed)
        condition = mx.concatenate((c, mx.array([[length / 31]], dtype=mx.float32)), axis=1)
        for i, time in enumerate(times):
            clean = mx.clip(model(x, mx.array([int(time) / 999]), condition), -1, 1)
            alpha = ALPHA_BAR[int(time)]
            following = ALPHA_BAR[int(times[i + 1])] if i + 1 < steps else mx.array(1., dtype=mx.float32)
            epsilon = (x - mx.sqrt(alpha) * clean) / mx.sqrt(1 - alpha)
            x = mx.where(payload, mx.sqrt(following) * clean + mx.sqrt(1 - following) * epsilon, fixed)
            mx.eval(x)
        images.append(np.asarray(mx.clip(x[0], -1, 1)))
        lengths.append(length)
    return np.stack(images), np.array(lengths, dtype=np.int32)


def sample(model, labels, keys, steps=25):
    if model.architecture == "G3-U":
        images, lengths = sample_images(model, labels, keys, steps)
        messages, margins, strict = codecs.decode(images, lengths, model.representation, model.src)
    else:
        tokens = sample_tokens(model, labels, keys)
        messages = data.decode(tokens, model.src)
        margins, strict = np.full(len(labels), np.nan), np.zeros(len(labels), dtype=bool)
    return messages, margins, strict


def score(model, clean, lengths, labels, keys):
    clean, lengths = mx.array(clean), mx.array(lengths)
    conditions = mx.array(condition_bits(labels))
    result = mx.zeros((len(labels),))
    for draw in range(8):
        if model.architecture == "G3-U":
            corruption = gaussian_corruption(clean, lengths, keys, model.representation, draw)
            result -= gaussian_loss_rows(model, clean, lengths, conditions, corruption) / 8
        else:
            result -= loss_rows(model, clean, lengths, conditions, corrupt(clean, lengths, keys, draw, model.src)) / 8
    values = np.asarray(result, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite CLP score")
    return values


def clp_pairs(model, clean, lengths, labels, pair_keys):
    if len(labels) % 2 or pair_keys.shape != (len(labels) // 2, 2):
        raise ValueError("CLP requires one explicit key per independent pair")
    keys = mx.repeat(pair_keys, 2, axis=0)
    donors = np.arange(len(labels)).reshape(-1, 2)[:, ::-1].reshape(-1)
    difference = score(model, clean, lengths, labels, keys) - score(model, clean, lengths, labels[donors], keys)
    return difference.reshape(-1, 2).sum(axis=1)
