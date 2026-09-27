"""MLX numerical and checkpoint operations for the shared study lifecycle."""
import io
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from . import mlx_models as models


def resolve_device(name):
    if name not in {"cpu", "gpu", "mps"}:
        raise ValueError("MLX device must be gpu or cpu")
    if name != "cpu" and not mx.metal.is_available():
        raise ValueError("MLX Metal GPU unavailable; no automatic CPU fallback")
    device = mx.Device(mx.cpu if name == "cpu" else mx.gpu)
    mx.set_default_device(device)
    return device


def clean_batch(encoder, rows, diffusion):
    array = np.stack([encoder.encode(bytes.fromhex(row[1])).numpy() for row in rows])
    clean = mx.array(array, dtype=mx.int32 if diffusion.discrete else mx.float32)
    if diffusion.discrete:
        diffusion.validate_clean(clean)
    else:
        clean = clean * 2 - 1
    return clean


def optimizer_and_step(model, diffusion, training):
    if training["weight_decay"] != 0:
        raise ValueError("The registered MLX Adam configuration requires zero weight decay")
    optimizer = optim.Adam(training["learning_rate"], betas=training["betas"], eps=training["eps"], bias_correction=True)
    optimizer.init(model.trainable_parameters())
    def objective(m, clean, cond, key):
        if getattr(diffusion, "factorized", False):
            return diffusion.loss(m, clean, cond, key=key, return_components=True)
        return diffusion.loss(m, clean, cond, key=key), {}

    value_and_grad = nn.value_and_grad(model, objective)

    def step(clean, cond, seed):
        (loss, components), gradients = value_and_grad(model, clean, cond, mx.random.key(seed))
        mx.eval(loss, components, gradients)
        if not models.finite([loss, components, gradients]):
            raise FloatingPointError("non-finite MLX training loss/gradient")
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        if not models.finite(model.parameters()):
            raise FloatingPointError("non-finite MLX model weights")
        return loss, components

    return optimizer, step


def validation_loss(p, stage, pipeline, model, diffusion, rows):
    from . import study_pilot as pilot
    source = p["pipelines"][pipeline]["source"]
    encoder = pilot.codecs(p["pipelines"][pipeline])[0]
    total = 0.
    model.eval()
    for start in range(0, len(rows), p["training"]["batch_size"]):
        batch = rows[start:start + p["training"]["batch_size"]]
        clean = clean_batch(encoder, batch, diffusion)
        cond = models.condition([row[0] for row in batch])
        for draw in range(1, p["training"]["validation_draws_per_condition"] + 1):
            keys = [mx.random.key(pilot.seed(p, stage, "validation-noise", source=source, pipeline=pipeline,
                                             unit_id=f"case:{row[0]:03x}", attempt=draw)) for row in batch]
            values = diffusion.losses(model, clean, cond, *diffusion.noise_inputs(clean, keys))
            if not models.finite(values):
                raise FloatingPointError("non-finite MLX validation loss")
            total += values.sum().item()
    return total / (len(rows) * p["training"]["validation_draws_per_condition"])


def checkpoint_bytes(state):
    tensors = dict(tree_flatten({name: state[name] for name in ("model", "optimizer")}))
    metadata = {name: value for name, value in state.items() if name not in {"model", "optimizer"}}
    # Corruption/generation have no implicit RNG: saved epoch/offset/update plus
    # frozen identity derive every explicit key again, including replayed work.
    metadata["rng_scheme"] = "explicit_mlx_key64_v1"
    stream = io.BytesIO()
    mx.save_safetensors(stream, tensors, metadata={"state": json.dumps(metadata, allow_nan=False)})
    return stream.getvalue()


def load_checkpoint(path):
    tensors, metadata = mx.load(path, return_metadata=True)
    state = json.loads(metadata["state"])
    if state.get("rng_scheme") != "explicit_mlx_key64_v1" or state["identity"].get("backend") != "mlx":
        raise ValueError("unsupported MLX checkpoint/RNG identity")
    state.update(tree_unflatten(list(tensors.items())))
    return state


def load_weights(model, parameters):
    model.load_weights(tree_flatten(parameters), strict=True)
    mx.eval(model.parameters())


def to_codec_tensor(value):
    # PyTorch is used only for the existing CPU codec boundary, never MLX math.
    import torch
    return torch.from_numpy(np.array(value, copy=True))


def profile_checks(p, pipeline, profile_id, model, diffusion, encoded):
    from . import study_pilot as pilot
    _, decoder, shape = pilot.codecs(p["pipelines"][pipeline])
    cond = models.condition([0, 4095])
    clean = mx.array(encoded[None].numpy(), dtype=mx.int32 if diffusion.discrete else mx.float32)
    if diffusion.discrete:
        diffusion.validate_clean(clean)
    else:
        clean = clean * 2 - 1
    model_cond = diffusion.payload_condition(cond[:1], diffusion.lengths(clean)) if profile_id == "D1" else cond[:1]
    output = model(clean, mx.zeros((1,)), model_cond)
    output_shape = (1, 32, diffusion.mask_token) if diffusion.discrete else clean.shape
    model_input_valid = models.finite(output) and output.shape == output_shape
    length_seeds = [181, 182] if profile_id == "D1" else None
    steps = p["model_profiles"][profile_id]["sampling_steps"]
    one = models.sample(model, diffusion, cond[:1], shape, steps=steps, seeds=[81],
                        length_seeds=length_seeds[:1] if length_seeds else None)
    repeat = models.sample(model, diffusion, cond[:1], shape, steps=steps, seeds=[81],
                           length_seeds=length_seeds[:1] if length_seeds else None)
    batch = models.sample(model, diffusion, cond, shape, steps=steps, seeds=[81, 82], length_seeds=length_seeds)
    one_np, batch_np = np.array(one), np.array(batch[:1])
    decode = lambda value: (decoder.decode(to_codec_tensor(value)) if diffusion.discrete
                            else decoder.decode(to_codec_tensor(value), normalized=True))
    a, b = decode(one[0]), decode(batch[0])
    decoder_equal = (a.valid, a.message, a.reason) == (b.valid, b.message, b.reason)
    repeat_exact = np.array_equal(one_np, np.array(repeat))
    batch_agrees = np.array_equal(one_np, batch_np) if diffusion.discrete else np.allclose(one_np, batch_np, atol=1e-3, rtol=1e-4)
    loss = diffusion.loss(model, clean, cond[:1], key=mx.random.key(83))
    parameterization = True
    if not diffusion.discrete:
        index = mx.array([diffusion.steps - 1])
        noise = mx.full(clean.shape, .125)
        noisy = diffusion.add_noise(clean, noise, index)
        predicted = diffusion.predicted_clean(noise if diffusion.prediction_type == "epsilon" else clean, noisy, index)
        parameterization = np.allclose(np.array(predicted), np.array(clean), atol=2e-5, rtol=2e-5)
    elif profile_id == "D1":
        parameterization = all(decode(row).valid for row in batch)
    return {"passed": bool(model_input_valid and models.finite(loss) and repeat_exact and batch_agrees and decoder_equal and parameterization),
            "model_input_valid": model_input_valid,
            "backend": "mlx", "same_key_repeat_exact": repeat_exact, "batch_numerically_equivalent": bool(batch_agrees),
            "batch_decoder_equal": decoder_equal, "training_parameterization": bool(parameterization),
            "batch_max_absolute_difference": float(np.max(np.abs(one_np - batch_np))),
            "parameters": sum(value.size for _, value in tree_flatten(model.parameters())),
            "sampling_nfe": 4 * p["model_profiles"][profile_id]["sampling_nfe_per_candidate"],
            "scope": "engineering_fixtures_not_generation_quality_qualification"}
