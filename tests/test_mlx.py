"""MLX math parity and native lifecycle checks; no formal PoC qualification."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from diffusion_hash_inv import mlx_models as models, mlx_backend as backend
from diffusion_hash_inv import study_pilot as pilot, study_cli


SPEC = Path(__file__).parents[1] / "examples/poc-v3.1-protocol.json"


def protocol():
    p = deepcopy(pilot.load_protocol(SPEC))
    p["synthetic"]["train_unique_messages"] = 16
    p["training"].update(batch_size=2, validation_draws_per_condition=1)
    p["pilot"]["P1"].update(train_messages=8, validation_conditions=2, evaluation_trials=2)
    p["execution"]["minimum_disk_free_gib"] = 0
    for name, spec in p["model_profiles"].items():
        spec.update(sampling_steps=2, sampling_nfe_per_candidate=3 if name == "D1" else 2)
        if spec["family"] == "gaussian":
            spec["diffusion_steps"] = 10
    return p


def import_reference_weights(model, reference):
    tensors = reference.state_dict()
    weights = []
    for name, _ in tree_flatten(model.parameters()):
        value = tensors[name.replace(".layers.", ".")].detach().numpy()
        if value.ndim == 4:
            value = value.transpose(1, 2, 3, 0) if name == "up.weight" else value.transpose(0, 2, 3, 1)
        weights.append((name, mx.array(np.ascontiguousarray(value))))
    model.load_weights(weights, strict=True)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("amendment", [None, "p2fix", "p2struct"])
def test_all_profiles_forward_loss_gradient_and_adam_match_reference(device, amendment):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("native Metal access required")
    backend.resolve_device(device)
    torch.set_num_threads(1)
    p = protocol()
    if amendment:
        revised = pilot.load_protocol(SPEC.with_name(f'poc-v3.1-{amendment}-protocol.json'))
        p['profile_selection'] = revised['profile_selection']
        p['model_profiles'] = deepcopy(revised['model_profiles'])
        for spec in p['model_profiles'].values():
            spec['sampling_steps'] = 2
            if spec['family'] == 'gaussian':
                spec['diffusion_steps'] = 10
    for pipeline in p["pipeline_order"]:
        cfg = p["pipelines"][pipeline]
        encoder, _, shape = pilot.codecs(cfg)
        source = cfg["source"]
        records = [b"000!", b"f" * 31] if source == "printable" else [b"\x00\x00\x00\x00", b"\x0f" * 31]
        clean_t = torch.stack([encoder.encode(row) for row in records])
        for profile in p["profile_selection"][cfg["model"] + "_order"]:
            reference, reference_diff = pilot.model_and_diffusion(p, pipeline, torch.device("cpu"), 11, profile_id=profile)
            model, diffusion = models.build_profile(p, pipeline, profile, seed=11)
            import_reference_weights(model, reference)
            assert sum(value.size for _, value in tree_flatten(model.parameters())) == sum(v.numel() for v in reference.parameters())
            cond_t = pilot.condition([0, 4095], torch.device("cpu"))
            cond = mx.array(cond_t.numpy())
            times_t = torch.tensor([.2, .8])
            times = mx.array(times_t.numpy())
            if not diffusion.discrete:
                x_t = clean_t.float() * 2 - 1
                x = mx.array(x_t.numpy())
                mc = diffusion.payload_condition(cond, diffusion.lengths(x)) if profile == 'G3' else cond
                tc = reference_diff.payload_condition(cond_t, reference_diff.lengths(x_t)) if profile == 'G3' else cond_t
                np.testing.assert_allclose(np.array(model(x, times, mc)), reference(x_t, times_t, tc).detach().numpy(), atol=3e-6, rtol=3e-5)
                indices_t = torch.tensor([1, 8])
                noise_t = torch.ones_like(x_t) * .125
                if getattr(reference_diff, 'loss_regions', None):
                    expected = reference_diff.losses(reference, x_t, cond_t, indices_t, noise_t)
                else:
                    noisy_t = reference_diff.add_noise(x_t, noise_t, indices_t)
                    target = noise_t if profile != 'G2' else x_t
                    expected = (reference(noisy_t, indices_t.float() / 9, cond_t) - target).square().flatten(1).mean(1)
                noise_inputs = mx.array([1, 8]), mx.array(noise_t.numpy())
            else:
                x = mx.array(clean_t.numpy(), dtype=mx.int32)
                diffusion.validate_clean(x)
                masks_t = torch.arange(32)[None].expand(2, -1) % 3 == 0
                masks = mx.array(masks_t.numpy())
                if profile == "D1":
                    expected, parts = reference_diff.losses(reference, clean_t, cond_t, times_t, masks_t, return_components=True)
                    native_values, native_parts = diffusion.losses(model, x, cond, times, masks, return_components=True)
                    np.testing.assert_allclose(np.array(native_values), expected.detach().numpy(), atol=4e-6, rtol=4e-5)
                    for name in parts:
                        np.testing.assert_allclose(np.array(native_parts[name]), parts[name].detach().numpy(), atol=4e-6, rtol=4e-5)
                else:
                    logits = reference(clean_t.masked_fill(masks_t, reference_diff.mask_token), times_t, cond_t)
                    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), clean_t, reduction="none")
                    expected = (losses * masks_t).sum(1) / masks_t.sum(1)
                noise_inputs = times, masks
            actual = diffusion.losses(model, x, cond, *noise_inputs)
            np.testing.assert_allclose(np.array(actual), expected.detach().numpy(), atol=4e-6, rtol=4e-5)
            _, gradients = nn.value_and_grad(model, lambda m: diffusion.losses(m, x, cond, *noise_inputs).mean())(model)
            expected.mean().backward()
            reference_params = dict(reference.named_parameters())
            shared_gradients = []
            for name, gradient in tree_flatten(gradients):
                expected_gradient = reference_params[name.replace(".layers.", ".")].grad.numpy()
                if expected_gradient.ndim == 4:
                    axes = (1, 2, 3, 0) if name == "up.weight" else (0, 2, 3, 1)
                    expected_gradient = expected_gradient.transpose(*axes)
                np.testing.assert_allclose(np.array(gradient), expected_gradient, atol=3e-6, rtol=1e-3)
                shared_gradients.append((name, mx.array(np.ascontiguousarray(expected_gradient))))
            optimizer, _ = backend.optimizer_and_step(model, diffusion, p["training"])
            # Isolate Adam parity: its epsilon denominator amplifies rounding
            # differences in nearly zero gradients, checked separately above.
            optimizer.update(model, tree_unflatten(shared_gradients))
            training = p["training"]
            torch.optim.Adam(reference.parameters(), lr=training["learning_rate"], betas=tuple(training["betas"]),
                             eps=training["eps"], foreach=False, fused=False).step()
            updated = {name: np.array(value) for name, value in tree_flatten(model.parameters())}
            import_reference_weights(model, reference)
            for name, value in tree_flatten(model.parameters()):
                np.testing.assert_allclose(updated[name], np.array(value), atol=4e-6, rtol=4e-5)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_native_mlx_training_sampling_and_recovery(tmp_path, monkeypatch, device):
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("native Metal access required")
    p = protocol()
    def forbidden(*args, **kwargs):
        raise AssertionError("MLX execution must not call PyTorch model/optimizer operations")
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    args = SimpleNamespace(workdir=tmp_path, stage="P0", device=device, backend="mlx", development=True, threads=1, resume=False)
    assert pilot.run_pilot(p, args)["status"] == "PASS"
    args.stage = "P1"
    assert pilot.run_pilot(p, args)["status"] == "PASS"
    summary = pilot.read_json(tmp_path / "pilot/P1/summary.json")
    assert summary["backend"] == "mlx" and len(summary["runs"]) == 28
    for name, row in summary["runs"].items():
        if name.startswith("random/"):
            continue
        directory = tmp_path / "pilot/P1/runs" / name
        state, _ = pilot.load_checkpoint(directory)
        assert state["identity"]["backend"] == "mlx" and state["update"] == 8
        assert state["rng_scheme"] == "explicit_mlx_key64_v1"
        assert not list((directory / "checkpoints").glob("*.pt"))
        assert list((directory / "checkpoints").glob("*.safetensors"))
        if name.endswith("/main"):
            assert row["recovery"]["status"] == "PASS"
        if row["profile_id"] == "D1":
            assert row["metrics"]["valid_rate"] == 1.
            records = pilot.logical_ledger(directory)
            assert all(r["length_rng_identity"] != r["rng_identity"] for r in records)
        assert all(r["backend"] == "mlx" for r in pilot.logical_ledger(directory))
    assert "Backend: mlx" in (tmp_path / "report.md").read_text()
    assert not pilot.write_report(tmp_path, p)["main_ready"]
    args.backend, args.device = "torch", "cpu"
    with pytest.raises(pilot.PilotError, match="environment"):
        pilot.run_pilot(p, args)


def test_cli_dry_run_and_invalid_backend_protocol_do_not_write(tmp_path, capsys):
    root = tmp_path / "uncreated"
    common = ["pilot", "--protocol", str(SPEC), "--workdir", str(root), "--backend", "mlx"]
    assert study_cli.main([*common, "--stage", "P1", "--development", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["backend"] == "mlx" and result["device"] == "gpu"
    assert study_cli.main([*common, "--stage", "P2", "--development"]) == 2
    assert not root.exists()


def test_key_boundary_empty_masks_and_nonfinite_outputs():
    backend.resolve_device("cpu")
    p = protocol()
    model, diffusion = models.build_profile(p, "P-DISC", "D1", seed=1)
    codec = pilot.codecs(p["pipelines"]["P-DISC"])[0]
    clean = mx.array(codec.encode(b"000!")[None].numpy(), dtype=mx.int32)
    cond = models.condition([0])
    model.length_head.weight = mx.zeros_like(model.length_head.weight)
    model.length_head.bias = mx.zeros_like(model.length_head.bias)
    losses = diffusion.losses(model, clean, cond, mx.zeros((1,)), mx.zeros((1, 32), dtype=mx.bool_))
    np.testing.assert_allclose(np.array(losses), [np.log(28)], atol=1e-6)
    with pytest.raises(ValueError, match="public"):
        models.condition([{"target": 0, "length": 4}])
    with pytest.raises(ValueError, match="separate length"):
        models.sample(model, diffusion, cond, (32,), steps=2, seeds=[1])
    invalid = np.array(clean)
    invalid[0, -1] = codec.eos
    with pytest.raises(ValueError, match="invalid D1"):
        diffusion.validate_clean(mx.array(invalid))
    assert not np.array_equal(np.array(mx.random.key(1)), np.array(mx.random.key(2**32 + 1)))
    model.length_head.bias = mx.full((28,), float("nan"))
    with pytest.raises(FloatingPointError, match="length"):
        models.sample(model, diffusion, cond, (32,), steps=2, seeds=[1], length_seeds=[2])
