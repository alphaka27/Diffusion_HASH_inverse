import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("PIL")

import mlx.core as mx
from PIL import Image

from diffusion_hash_inv.models.conditional_diffusion_mlx import (
    MLXConditionalDiffusionTrainConfig,
    MLXConditionalDenoiser,
    MLXDDPMScheduler,
    MLXGeneratedImageDataset,
    discover_generated_image_samples_mlx,
    train_conditional_diffusion_mlx,
)


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


def test_train_conditional_diffusion_mlx_smoke(tmp_path: Path) -> None:
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
        columns=2,
        log_every=1,
    )

    output_path = train_conditional_diffusion_mlx(config)

    assert output_path.is_file()
    assert (config.output_dir / "config.json").is_file()
    assert (config.output_dir / "label_map.json").is_file()
