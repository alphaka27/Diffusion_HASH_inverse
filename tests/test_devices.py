"""Backend selection and real training/checkpoint/sampling checks."""

import json
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from diffusion_hash_inv.dataset import build_digest_records
from diffusion_hash_inv.devices import resolve_device, synchronize
from diffusion_hash_inv.experiment_cli import main
from diffusion_hash_inv.runner import ExperimentConfig, _build_model, _codec, _train


def test_device_selection_and_explicit_gpu_failure():
    with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
        assert str(resolve_device("auto")) == "mps"
        assert str(resolve_device("cpu")) == "cpu"
        assert str(resolve_device("mps:0")) == "mps"
    with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
        assert str(resolve_device("auto")) == "cpu"
        with pytest.raises(RuntimeError, match="sandbox"):
            resolve_device("mps")
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
        assert str(resolve_device("auto")) == "cuda:0"
        assert str(resolve_device("cuda:1")) == "cuda:1"
        with pytest.raises(ValueError, match="does not exist"):
            resolve_device("cuda:2")
    for value in ("mps:1", "cpu:2", "meta", "nonsense"):
        with pytest.raises(ValueError):
            resolve_device(value)


@pytest.mark.parametrize("backend", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("representation,source", [
    ("bgv", "printable"), ("cgge", "printable"), ("tokens", "printable"),
    ("bgv", "random_bytes"), ("tokens", "random_bytes"),
])
def test_training_checkpoint_and_sampling_on_backend(tmp_path, backend, representation, source):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available to this process")
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available to this process")
    torch.set_num_threads(1)
    device = resolve_device(backend)
    options = dict(masking_schedule=(0, .5, 1), token_embedding_dim=2, token_temperature=1.) if representation == "tokens" else {}
    config = ExperimentConfig(representation, source, "md5", 8, condition_format="canonical_bits",
                              condition_dim=259, training_steps=2, batch_size=2, width=8,
                              diffusion_steps=4, sampling_steps=2, device=str(device), **options)
    records = build_digest_records((b"ABCD", b"EFGHI"), source=source, algorithm="md5", q=8)
    encoder, decoder, shape = _codec(representation, source=source)
    torch.manual_seed(7)
    model = _build_model(config, shape).to(device)
    before = [p.detach().cpu().clone() for p in model.parameters()]
    checkpoint = tmp_path / "resume.pt"
    diffusion, loss = _train(model, records, config, encoder, device, checkpoint_path=checkpoint)
    assert torch.isfinite(torch.tensor(loss))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(a, b.cpu()) for a, b in zip(before, model.parameters()))
    restored = _build_model(config, shape).to(device)
    with patch("torch.optim.Adam.step", side_effect=AssertionError("completed training must not repeat")):
        _, restored_loss = _train(restored, records, config, encoder, device, checkpoint_path=checkpoint)
    assert loss == restored_loss
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    model.eval()
    condition = torch.zeros((2, 259), device=device)
    sampling = dict(temperature=1.) if representation == "tokens" else {}
    generator = torch.Generator(device=device).manual_seed(8)
    state = generator.get_state()
    sample = diffusion.sample(model, condition, shape, sampling_steps=2, generator=generator, **sampling)
    generator.set_state(state)
    replay = diffusion.sample(restored.eval(), condition, shape, sampling_steps=2, generator=generator, **sampling)
    synchronize(device)
    assert sample.device.type == device.type and torch.isfinite(sample).all()
    assert sample.shape == (2, *shape) and torch.equal(sample, replay)
    # Decoding accepts arbitrary generated output; validity is not a smoke-test requirement.
    decoder.decode(sample[0].cpu() if representation == "tokens" else (sample[0].cpu() + 1) / 2)


def test_cli_device_override_is_frozen_as_actual_backend(tmp_path, capsys):
    config = ExperimentConfig("bits", "printable", "md5", 8, dataset_size=12, method="random", device="mps")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "run"
    assert main(["--config", str(path), "--output", str(output), "--device", "cpu"]) == 0
    capsys.readouterr()
    assert json.loads((output / "configuration_frozen.json").read_text())["config"]["device"] == "cpu"
