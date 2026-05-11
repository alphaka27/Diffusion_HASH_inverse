import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image

from diffusion_hash_inv.models.loop_conditioned_diffusion import (
    LoopConditionedDDPMScheduler,
    LoopConditionedDiffusionTrainConfig,
    LoopConditionedImageDataset,
    LoopConditionedUNet,
    extract_loop_condition,
    loop_key,
    train_loop_conditioned_diffusion,
)


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def _step_payload(seed: int = 0, loop_count: int = 64) -> dict[str, object]:
    loops: dict[str, object] = {}
    for idx in range(1, loop_count + 1):
        base = seed + idx
        loops[loop_key(idx)] = {
            "A": f"0x{base:08x}",
            "B": f"0x{base + 1:08x}",
            "C": f"0x{base + 2:08x}",
            "D": f"0x{base + 3:08x}",
        }
    return {"Logs": {"4th Step": {"1st Round": loops}}}


def _write_loop_dataset(tmp_path: Path, count: int = 2) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    for idx in range(count):
        run_id = f"RUN_{idx:04d}"
        _write_png(image_root / run_id / "message.png", (8, 8), (idx, 2, 3, 255))
        json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(_step_payload(seed=idx * 100)), encoding="utf-8")
    return image_root, json_root


def test_extract_loop_condition_uses_all_64_loops() -> None:
    payload = _step_payload()
    condition = extract_loop_condition(payload)

    assert condition.shape == (64, 4)
    assert condition.dtype.name == "float32"
    assert condition.min() >= -1.0
    assert condition.max() <= 1.0


def test_extract_loop_condition_normalizes_uint32_words() -> None:
    loops = {
        loop_key(idx): {"A": "0x00000000", "B": "0xffffffff", "C": 0, "D": 0xFFFFFFFF}
        for idx in range(1, 65)
    }
    payload = {"Logs": {"4th Step": {"1st Round": loops}}}

    condition = extract_loop_condition(payload)

    assert condition[0].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_loop_conditioned_dataset_returns_condition_tensor(tmp_path: Path) -> None:
    image_root, json_root = _write_loop_dataset(tmp_path)

    dataset = LoopConditionedImageDataset(
        image_root,
        json_root=json_root,
        image_size=8,
        fit_mode="resize",
    )
    image, condition, index = dataset[0]

    assert len(dataset) == 2
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert condition.shape == (64, 4)
    assert condition.dtype == torch.float32
    assert index.dtype == torch.long


def test_loop_conditioned_unet_and_scheduler_preserve_shape() -> None:
    model = LoopConditionedUNet(
        in_channels=3,
        condition_shape=(64, 4),
        base_channels=4,
        time_dim=8,
    )
    scheduler = LoopConditionedDDPMScheduler(timesteps=2)
    x0 = torch.randn(2, 3, 8, 8)
    conditions = torch.randn(2, 64, 4)
    timesteps = torch.tensor([0, 1], dtype=torch.long)
    noise = torch.randn_like(x0)

    xt = scheduler.q_sample(x0, timesteps, noise)
    pred = model(xt, timesteps, conditions)
    samples = scheduler.sample(model, (2, 3, 8, 8), conditions)

    assert pred.shape == x0.shape
    assert samples.shape == x0.shape


def test_loop_conditioned_training_smoke_run(tmp_path: Path) -> None:
    image_root, json_root = _write_loop_dataset(tmp_path)
    output_dir = tmp_path / "loop_conditioned"
    config = LoopConditionedDiffusionTrainConfig(
        data_root=image_root,
        json_root=json_root,
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

    result = train_loop_conditioned_diffusion(config)

    assert result["dataset_size"] == 2
    assert result["condition_rows"] == 64
    assert result["condition_columns"] == 4
    assert Path(result["checkpoint"]).exists()
    assert Path(result["sample_grid"]).exists()
    assert (output_dir / "condition_schema.json").exists()
