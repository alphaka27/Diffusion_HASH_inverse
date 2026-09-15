"""Direct-bit and tiny end-to-end runner checks."""

import json
from contextlib import redirect_stdout
from io import StringIO

import torch
import pytest

from diffusion_hash_inv.dataset import build_digest_records
from diffusion_hash_inv.encoding import DirectBitsDecoder, DirectBitsEncoder
from diffusion_hash_inv.experiment_cli import main as experiment_main
from diffusion_hash_inv.runner import ExperimentConfig, _conditions, run_experiment
from diffusion_hash_inv.models import ImageUNet


def test_direct_bits_round_trip_all_boundary_lengths() -> None:
    encoder, decoder = DirectBitsEncoder(), DirectBitsDecoder()
    for length in range(4, 32):
        message = bytes(range(length))
        result = decoder.decode(encoder.encode(message) * 2 - 1, normalized=True)
        assert result.valid and result.message == message and result.length == length


def test_runner_writes_a_tiny_direct_bits_diffusion_run(tmp_path) -> None:
    config = ExperimentConfig(
        representation="bits",
        source="printable",
        algorithm="md5",
        q=8,
        dataset_size=12,
        k=1,
        training_steps=1,
        batch_size=2,
        diffusion_steps=2,
        sampling_steps=2,
        width=4,
    )
    result = run_experiment(config, tmp_path)
    assert result.parameter_count and result.training_loss is not None
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "evaluation" / "metrics.json").is_file()
    assert (tmp_path / "evaluation" / "metrics.csv").is_file()
    assert json.loads((tmp_path / "run_manifest.json").read_text())["representation"] == "bits"


def test_predictor_and_image_unet_controls_share_the_runner_shapes(tmp_path) -> None:
    for width in (64, 128):
        model = ImageUNet(2, condition_dim=8, width=4)
        assert model(torch.zeros((1, 2, 32, width)), torch.zeros(1), torch.zeros((1, 8))).shape == (1, 2, 32, width)
    config = ExperimentConfig(
        representation="bits",
        source="random_bytes",
        algorithm="sha256",
        q=8,
        dataset_size=12,
        k=1,
        method="predictor",
        condition_mode="zero",
        training_steps=1,
        batch_size=2,
        diffusion_steps=2,
        sampling_steps=2,
        width=4,
    )
    assert run_experiment(config, tmp_path).training_loss is not None


def test_shuffled_model_condition_is_also_shuffled_at_evaluation() -> None:
    records = build_digest_records((b"ABCD", b"EFGH"), source="printable", algorithm="md5", q=8)
    config = ExperimentConfig(representation="bits", source="printable", algorithm="md5", q=8, condition_mode="shuffled_hash")
    shuffled = _conditions(records, config)
    assert torch.equal(shuffled[0], _conditions(records, ExperimentConfig(representation="bits", source="printable", algorithm="md5", q=8))[1])
    with pytest.raises(ValueError, match="K=1"):
        ExperimentConfig(representation="bits", source="printable", algorithm="md5", q=8, method="predictor", k=2)


def test_experiment_cli_runs_a_json_config(tmp_path) -> None:
    config = {
        "representation": "bits",
        "source": "printable",
        "algorithm": "md5",
        "q": 8,
        "dataset_size": 12,
        "method": "random",
        "training_steps": 1,
        "batch_size": 1,
        "diffusion_steps": 2,
        "sampling_steps": 1,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    stdout = StringIO()
    with redirect_stdout(stdout):
        assert experiment_main(["--config", str(config_path), "--output", str(tmp_path / "run")]) == 0
    assert json.loads(stdout.getvalue())["summary"]["candidate_budget"] == 1
