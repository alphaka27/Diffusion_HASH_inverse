import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("PIL")

import mlx.core as mx
from PIL import Image

import diffusion_hash_inv.models.conditional_diffusion_mlx as conditional_diffusion_mlx_module
from diffusion_hash_inv.models.conditional_diffusion_mlx import (
    MLXConditionalDiffusionTrainConfig,
    MLXConditionalDenoiser,
    MLXDDPMScheduler,
    MLXGeneratedImageDataset,
    _image_from_vector,
    build_beta_schedule_mlx,
    discover_generated_image_samples_mlx,
    train_conditional_diffusion_mlx,
)
from diffusion_hash_inv.models.sample_decoding import _byte2rgb_decoder


def _write_png(path: Path, size: tuple[int, int], color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color).save(path)


def _write_json(json_root: Path, run_id: str, final_hash: str) -> None:
    path = json_root / "2026-05-12 12-00-00" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"Message": {"Hex": "0x00"}, "Generated hash": final_hash}),
        encoding="utf-8",
    )


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "data" / "images"
    json_root = tmp_path / "output" / "json"
    _write_png(image_root / "RUN_0001" / "message.png", (8, 4), 32)
    _write_png(image_root / "RUN_0002" / "message.png", (4, 8), 224)
    _write_json(json_root, "RUN_0001", "0xaaa")
    _write_json(json_root, "RUN_0002", "0xbbb")
    return image_root, json_root


def test_discover_generated_image_samples_mlx_uses_final_hash_labels(tmp_path: Path) -> None:
    image_root, json_root = _write_dataset(tmp_path)

    samples, condition_to_idx = discover_generated_image_samples_mlx(
        image_root,
        json_root=json_root,
    )

    assert len(samples) == 2
    assert set(condition_to_idx) == {"0xaaa", "0xbbb"}
    assert {sample.condition for sample in samples} == {"0xaaa", "0xbbb"}


def test_mlx_dataset_batches_flattened_normalized_images(tmp_path: Path) -> None:
    image_root, json_root = _write_dataset(tmp_path)
    dataset = MLXGeneratedImageDataset(
        image_root,
        json_root=json_root,
        image_size=8,
        channels=1,
    )

    images, labels = dataset.batch(mx.array([0, 1]).astype(mx.int32))

    assert images.shape == (2, 64)
    assert labels.shape == (2,)
    assert dataset.num_conditions == 2
    assert float(mx.min(images)) >= -1.0
    assert float(mx.max(images)) <= 1.0


def test_mlx_dataset_supports_height_flatten_one_pixel_blocks(tmp_path: Path) -> None:
    image_root = tmp_path / "data" / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "RUN_0001"
    image_path = image_root / run_id / "message.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    source = Image.new("L", (112, 28), 255)
    block_values = [0, 32, 128, 255]
    for index, value in enumerate(block_values):
        source.paste(Image.new("L", (28, 28), value), (index * 28, 0))
    source.save(image_path)
    _write_json(json_root, run_id, "0xaaa")

    dataset = MLXGeneratedImageDataset(
        image_root,
        json_root=json_root,
        image_size=8,
        channels=1,
        fit_mode="height-flatten",
    )
    image, label = dataset[0]

    assert label == 0
    assert dataset.output_image_size == 2
    assert dataset.image_dim == 4
    assert image.shape == (4,)
    expected = [(value / 127.5) - 1.0 for value in block_values]
    assert image.tolist() == pytest.approx(expected, abs=1e-6)


def test_mlx_dataset_supports_cube_id_grid_rgb_blocks(tmp_path: Path) -> None:
    image_root = tmp_path / "data" / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "RUN_0001"
    image_path = image_root / run_id / "message.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = _byte2rgb_decoder("cube-id")
    payload = b"\x00\x7f\x80\xff"
    encoded = decoder.rgb_encoder(payload)
    pixels = encoded if isinstance(encoded, tuple) else (encoded,)
    block_colors = [pixel.as_tuple for pixel in pixels]
    source = Image.new("RGB", (112, 28))
    for index, color in enumerate(block_colors):
        source.paste(Image.new("RGB", (28, 28), color), (index * 28, 0))
    source.save(image_path)
    _write_json(json_root, run_id, "0xaaa")

    dataset = MLXGeneratedImageDataset(
        image_root,
        json_root=json_root,
        image_size=8,
        channels=3,
        fit_mode="cube-id-grid",
    )
    image, label = dataset[0]

    assert label == 0
    assert dataset.output_image_size == 2
    assert dataset.image_dim == 12
    assert image.shape == (12,)
    expected = np.asarray(block_colors, dtype=np.float32).reshape(2, 2, 3)
    expected = (expected / 127.5 - 1.0).transpose(2, 0, 1).reshape(-1)
    assert image.tolist() == pytest.approx(expected.tolist(), abs=1e-6)

    with pytest.raises(ValueError, match="cube-id-grid fit mode requires channels=3"):
        MLXGeneratedImageDataset(
            image_root,
            json_root=json_root,
            image_size=8,
            channels=1,
            fit_mode="cube-id-grid",
        )


def test_mlx_image_from_vector_rounds_normalized_uint8_values() -> None:
    original = np.array([2, 38, 119, 212], dtype=np.float32)
    vector = original / 127.5 - 1.0

    image = _image_from_vector(vector, image_size=2, channels=1)

    assert np.asarray(image, dtype=np.uint8).reshape(-1).tolist() == [2, 38, 119, 212]


def test_mlx_scheduler_and_denoiser_preserve_shape() -> None:
    scheduler = MLXDDPMScheduler(timesteps=4)
    model = MLXConditionalDenoiser(
        image_dim=16,
        num_conditions=2,
        time_dim=8,
        hidden_dim=16,
    )
    x = mx.zeros((2, 16), dtype=mx.float32)
    labels = mx.array([0, 1], dtype=mx.int32)

    sampled = scheduler.p_sample(model, x, step=3, labels=labels)

    assert sampled.shape == x.shape


def test_mlx_scheduler_accepts_custom_betas() -> None:
    scheduler = MLXDDPMScheduler(betas=[0.01, 0.02, 0.03])

    assert scheduler.timesteps == 3
    assert np.asarray(scheduler.betas).tolist() == pytest.approx([0.01, 0.02, 0.03])


def test_build_beta_schedule_mlx_linear_auto_uses_hash_approach_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyAnalyze:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def summarize_beta_schedules(self, **_kwargs: object) -> object:
            return SimpleNamespace(mean=[1.0, 2.0, 3.0, 4.0])

    class DummyBetaScheduler:
        def __init__(self, beta_min: float, beta_max: float) -> None:
            self.beta_min = beta_min
            self.beta_max = beta_max

        def approach1(self, _mean: object) -> object:
            return SimpleNamespace(rescaled_candidate=[0.11, 0.12, 0.13, 0.14])

        def approach2(self, _mean: object) -> object:
            return SimpleNamespace(candidate=[0.21, 0.22, 0.23, 0.24])

    monkeypatch.setattr(conditional_diffusion_mlx_module, "Analyze", DummyAnalyze)
    monkeypatch.setattr(conditional_diffusion_mlx_module, "BetaScheduler", DummyBetaScheduler)
    config = MLXConditionalDiffusionTrainConfig(
        beta_schedule="linear",
        timesteps="auto",
        beta_start=0.001,
        beta_end=0.009,
    )

    betas = build_beta_schedule_mlx(config)

    assert betas is not None
    assert betas.shape == (4,)
    assert betas.tolist() == pytest.approx([0.001, 0.0036666667, 0.0063333333, 0.009])


def test_build_beta_schedule_mlx_uses_hash_approach_betas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyAnalyze:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def summarize_beta_schedules(self, **_kwargs: object) -> object:
            return SimpleNamespace(mean=[1.0, 2.0, 3.0])

    class DummyBetaScheduler:
        def __init__(self, beta_min: float, beta_max: float) -> None:
            self.beta_min = beta_min
            self.beta_max = beta_max

        def approach1(self, _mean: object) -> object:
            return SimpleNamespace(rescaled_candidate=[0.11, 0.12, 0.13])

        def approach2(self, _mean: object) -> object:
            return SimpleNamespace(candidate=[0.21, 0.22, 0.23])

    monkeypatch.setattr(conditional_diffusion_mlx_module, "Analyze", DummyAnalyze)
    monkeypatch.setattr(conditional_diffusion_mlx_module, "BetaScheduler", DummyBetaScheduler)

    approach1 = build_beta_schedule_mlx(
        MLXConditionalDiffusionTrainConfig(beta_schedule="hash-approach1")
    )
    approach2 = build_beta_schedule_mlx(
        MLXConditionalDiffusionTrainConfig(beta_schedule="hash-approach2")
    )

    assert approach1 is not None
    assert approach2 is not None
    assert approach1.tolist() == pytest.approx([0.11, 0.12, 0.13])
    assert approach2.tolist() == pytest.approx([0.21, 0.22, 0.23])


def test_train_conditional_diffusion_mlx_smoke(
    tmp_path: Path,
) -> None:
    image_root, json_root = _write_dataset(tmp_path)
    config = MLXConditionalDiffusionTrainConfig(
        data_root=image_root,
        json_root=json_root,
        output_dir=tmp_path / "out",
        image_size=8,
        channels=1,
        batch_size=2,
        train_steps=1,
        timesteps=4,
        time_dim=8,
        hidden_dim=16,
        sample_count=2,
        save_process_traces=True,
        trace_sample_count=1,
        columns=2,
        log_every=1,
    )

    expected_sample_dir = config.output_dir / "sample"
    expected_source_dir = expected_sample_dir / "source"
    expected_final_dir = expected_sample_dir / "final"
    stale_sample_dir = expected_sample_dir
    stale_sample_dir.mkdir(parents=True)
    for stale_name in ("preview.png", "source.png", "final.png", "source_999.png", "final_999.png"):
        (stale_sample_dir / stale_name).write_bytes(b"stale")
    expected_source_dir.mkdir(parents=True)
    expected_final_dir.mkdir(parents=True)
    (expected_source_dir / "source_999.png").write_bytes(b"stale")
    (expected_final_dir / "final_999.png").write_bytes(b"stale")

    output_path = train_conditional_diffusion_mlx(config)

    assert output_path.is_file()
    assert output_path.name == "final.labels.json"
    assert output_path.parent == expected_final_dir
    assert (expected_source_dir / "source.labels.json").is_file()
    assert not (expected_sample_dir / "preview.png").exists()
    assert not (expected_sample_dir / "preview.labels.json").exists()
    assert not (expected_sample_dir / "source.png").exists()
    assert not (expected_sample_dir / "final.png").exists()
    assert not (expected_sample_dir / "source_999.png").exists()
    assert not (expected_sample_dir / "final_999.png").exists()
    assert not (expected_source_dir / "source_999.png").exists()
    assert not (expected_final_dir / "final_999.png").exists()
    comparison_path = expected_sample_dir / "decode_comparison.json"
    assert comparison_path.is_file()
    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison_payload["total"] == 2
    assert len(comparison_payload["records"]) == 2
    assert comparison_payload["records"][0]["source"]["supported"] is False
    for index in range(config.sample_count):
        with Image.open(expected_source_dir / f"source_{index:03d}.png") as source_image:
            assert source_image.size == (8, 8)
        with Image.open(expected_final_dir / f"final_{index:03d}.png") as final_image:
            assert final_image.size == (8, 8)
    assert (config.output_dir / "config.json").is_file()
    assert (config.output_dir / "label_map.json").is_file()
    beta_schedule = json.loads((config.output_dir / "beta_schedule.json").read_text())
    assert beta_schedule["mode"] == "linear"
    assert beta_schedule["timesteps"] == 4
    assert len(beta_schedule["betas"]) == 4

    checkpoint_dir = config.output_dir / "checkpoints"
    metadata_path = checkpoint_dir / "step_000001.json"
    model_weights_path = checkpoint_dir / "step_000001.safetensors"
    optimizer_state_path = checkpoint_dir / "step_000001.optimizer.safetensors"
    assert metadata_path.is_file()
    assert model_weights_path.is_file()
    assert optimizer_state_path.is_file()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["step"] == 1
    assert metadata["model_weights"] == model_weights_path.name
    assert metadata["optimizer_state"] == optimizer_state_path.name

    trace_root = config.output_dir / "process_traces"
    assert (trace_root / "forward" / "png" / "x0.png").is_file()
    assert (trace_root / "forward" / "png" / "t_000000.png").is_file()
    assert (trace_root / "forward" / "png" / "t_000003.png").is_file()
    assert (trace_root / "forward" / "json" / "x0.labels.json").is_file()
    assert (trace_root / "reverse" / "png" / "xT_noise.png").is_file()
    assert (trace_root / "reverse" / "png" / "t_000003.png").is_file()
    assert (trace_root / "reverse" / "png" / "t_000000.png").is_file()
    assert (trace_root / "reverse" / "json" / "xT_noise.labels.json").is_file()
    assert len(list((trace_root / "forward" / "png").glob("t_*.png"))) == 4
    assert len(list((trace_root / "forward" / "json").glob("t_*.labels.json"))) == 4
    assert len(list((trace_root / "reverse" / "png").glob("t_*.png"))) == 4
    assert len(list((trace_root / "reverse" / "json").glob("t_*.labels.json"))) == 4
