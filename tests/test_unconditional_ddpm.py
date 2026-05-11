from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image

from diffusion_hash_inv.models.unconditional_ddpm import (
    UnconditionalDDPMTrainConfig,
    UnconditionalDDPMScheduler,
    UnconditionalImageDataset,
    UnconditionalUNet,
    discover_unconditional_image_samples,
    train_unconditional_ddpm,
)


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def _write_unconditional_dataset(tmp_path: Path, count: int = 2) -> Path:
    image_root = tmp_path / "images"
    for idx in range(count):
        run_id = f"RUN_{idx:04d}"
        _write_png(image_root / run_id / "message.png", (8, 8), (idx, 2, 3, 255))
        _write_png(image_root / run_id / "4th Step" / "1st Round.png", (8, 8), (9, 9, 9, 255))
    return image_root


def test_discover_unconditional_samples_uses_only_message_png(tmp_path: Path) -> None:
    image_root = _write_unconditional_dataset(tmp_path)

    samples = discover_unconditional_image_samples(image_root)

    assert len(samples) == 2
    assert all(sample.path.name == "message.png" for sample in samples)
    assert {sample.run_id for sample in samples} == {"RUN_0000", "RUN_0001"}


def test_unconditional_dataset_returns_image_and_index(tmp_path: Path) -> None:
    image_root = _write_unconditional_dataset(tmp_path)

    dataset = UnconditionalImageDataset(
        image_root,
        image_size=8,
        fit_mode="resize",
    )
    image, index = dataset[0]

    assert len(dataset) == 2
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert index.dtype == torch.long


def test_unconditional_unet_and_scheduler_preserve_shape() -> None:
    model = UnconditionalUNet(in_channels=3, base_channels=4, time_dim=8)
    scheduler = UnconditionalDDPMScheduler(timesteps=2)
    x0 = torch.randn(2, 3, 8, 8)
    timesteps = torch.tensor([0, 1], dtype=torch.long)
    noise = torch.randn_like(x0)

    xt = scheduler.q_sample(x0, timesteps, noise)
    pred = model(xt, timesteps)
    samples = scheduler.sample(model, (2, 3, 8, 8), device="cpu")

    assert pred.shape == x0.shape
    assert samples.shape == x0.shape


def test_unconditional_training_smoke_run(tmp_path: Path) -> None:
    image_root = _write_unconditional_dataset(tmp_path)
    output_dir = tmp_path / "unconditional"
    config = UnconditionalDDPMTrainConfig(
        data_root=image_root,
        output_dir=output_dir,
        image_size=8,
        fit_mode="resize",
        max_images=2,
        batch_size=1,
        train_steps=1,
        timesteps=2,
        base_channels=4,
        time_dim=8,
        sample_count=1,
        device="cpu",
    )

    result = train_unconditional_ddpm(config)

    assert result["dataset_size"] == 2
    assert result["train_steps"] == 1
    assert Path(result["checkpoint"]).exists()
    assert Path(result["sample_grid"]).exists()
    train_config = (output_dir / "train_config.json").read_text(encoding="utf-8")
    assert "label_source" not in train_config
