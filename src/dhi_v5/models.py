"""Fresh MLX-only masked diffusion models and identity-keyed samplers."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .data import EOS, MASK, PAD, VOCAB, WIDTH, condition_bits, key_words, seed


class Denoiser(nn.Module):
    def __init__(self, architecture="D1-S"):
        super().__init__()
        self.architecture = architecture
        self.length_head = nn.Linear(12, 28)
        if architecture == "D1-S":
            self.embedding = nn.Embedding(VOCAB, 16)
            self.hidden = nn.Linear(WIDTH * 16 + 14, 128)
            self.output = nn.Linear(128, WIDTH * 96)
            self.residual = nn.Linear(13, WIDTH * 96)
        elif architecture in ("D1-T", "D1-T-L"):
            dim, layers, heads = (192, 4, 4) if architecture == "D1-T" else (256, 8, 8)
            self.embedding = nn.Embedding(VOCAB, dim)
            self.position = nn.Embedding(WIDTH, dim)
            self.context = nn.Linear(14, dim)
            self.blocks = [nn.TransformerEncoderLayer(dim, heads, 4 * dim, activation=nn.gelu, norm_first=True) for _ in range(layers)]
            self.norm = nn.LayerNorm(dim)
            self.output = nn.Linear(dim, 96)
            self.residual = nn.Linear(13, WIDTH * 96)
        else:
            raise ValueError("Architecture must be D1-S, D1-T or D1-T-L")

    def __call__(self, tokens, time, condition, lengths):
        c = mx.concatenate((condition, lengths[:, None].astype(mx.float32) / 31), axis=1)
        context = mx.concatenate((c, time[:, None]), axis=1)
        # Dense one-hot multiplication avoids nondeterministic Metal scatter-add
        # in embedding gradients, so interrupted training reproduces bitwise.
        h = (tokens[:, :, None] == mx.arange(VOCAB)).astype(mx.float32) @ self.embedding.weight
        if self.architecture == "D1-S":
            h = nn.silu(self.hidden(mx.concatenate((h.reshape(len(tokens), -1), context), axis=1)))
            logits = self.output(h).reshape(-1, WIDTH, 96)
        else:
            h = h + self.position(mx.arange(WIDTH))[None, :, :] + self.context(context)[:, None, :]
            padding = mx.arange(WIDTH)[None, :] > lengths[:, None]
            attention_mask = mx.where(padding[:, None, None, :], -1e9, 0.0)
            for block in self.blocks:
                h = block(h, attention_mask)
            logits = self.output(self.norm(h))
        # EOS/PAD have model logits but are placed only by the sampled length.
        return (logits + self.residual(c).reshape(-1, WIDTH, 96))[:, :, :94]


def make_model(architecture, namespace):
    mx.random.seed(seed("weights", *namespace) & 0xFFFFFFFF)
    model = Denoiser(architecture)
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


def corrupt(tokens, lengths, keys, draw=0):
    time = mx.vmap(lambda k: mx.random.uniform(key=k))(fork_keys(keys, 2 * draw))
    uniforms = mx.vmap(lambda k: mx.random.uniform(shape=(WIDTH,), key=k))(fork_keys(keys, 2 * draw + 1))
    masked = (uniforms < time[:, None]) & (mx.arange(WIDTH)[None, :] < lengths[:, None])
    return mx.where(masked, MASK, tokens), time, masked


def loss_rows(model, tokens, lengths, conditions, corruption):
    noisy, time, masked = corruption
    logits = model(noisy, time, conditions, lengths)
    ce = nn.losses.cross_entropy(logits, mx.clip(tokens, 0, 93), reduction="none")
    payload = mx.sum(ce * masked, axis=1) / mx.maximum(mx.sum(masked, axis=1), 1)
    return nn.losses.cross_entropy(model.length_head(conditions), lengths - 4, reduction="none") + payload


def objective(model, tokens, lengths, conditions, corruption):
    return loss_rows(model, tokens, lengths, conditions, corruption).mean()


def train_step(model, optimizer, tokens, lengths, labels, keys, lr, update):
    tokens, lengths = mx.array(tokens), mx.array(lengths)
    conditions = mx.array(condition_bits(labels))
    corruption = corrupt(tokens, lengths, keys)
    loss, grads = nn.value_and_grad(model, objective)(model, tokens, lengths, conditions, corruption)
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


def sample(model, labels, keys):
    """32 reverse intervals, one sampled length, no remasking; NFE=33."""
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


def sample_reference(model, labels, keys):
    """New scalar oracle for the V5 rewrite gate, never calls a legacy sampler."""
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


def score(model, tokens, lengths, labels, keys):
    tokens, lengths = mx.array(tokens), mx.array(lengths)
    conditions = mx.array(condition_bits(labels))
    result = mx.zeros((len(labels),))
    for draw in range(8):
        result -= loss_rows(model, tokens, lengths, conditions, corrupt(tokens, lengths, keys, draw)) / 8
    return np.asarray(result, dtype=np.float64)


def clp_pairs(model, tokens, lengths, labels, keys):
    if len(labels) % 2:
        raise ValueError("CLP requires independent pairs")
    donors = np.arange(len(labels)).reshape(-1, 2)[:, ::-1].reshape(-1)
    difference = score(model, tokens, lengths, labels, keys) - score(model, tokens, lengths, labels[donors], keys)
    return difference.reshape(-1, 2).sum(axis=1)
