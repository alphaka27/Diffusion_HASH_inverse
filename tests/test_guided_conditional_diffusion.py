import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image

from diffusion_hash_inv.models.conditional_diffusion import ConditionalUNet, DDPMNoiseScheduler
from diffusion_hash_inv.models.guided_conditional_diffusion import (
    GuidedConditionalDiffusionTrainConfig,
    NoisyImageClassifier,
    apply_condition_dropout,
    sample_classifier_free_guidance,
    sample_classifier_guidance,
    train_guided_conditional_diffusion,
)


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def _write_guidance_dataset(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    for idx, color in enumerate([(255, 0, 0, 255), (0, 0, 255, 255)]):
        run_id = f"RUN_{idx:04d}"
        _write_png(image_root / run_id / "message.png", (8, 8), color)
        json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "Message": {"Hex": f"0xmessage{idx}"},
                    "Logs": {"4th Step": f"0xstep4-{idx}"},
                    "Generated hash": f"0xfinal-{idx}",
                    "Correct   hash": f"0xfinal-{idx}",
                }
            ),
            encoding="utf-8",
        )
    return image_root, json_root


def test_apply_condition_dropout_can_replace_all_labels() -> None:
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    dropped = apply_condition_dropout(labels, null_label=3, dropout=1.0)

    assert dropped.tolist() == [3, 3, 3]


def test_classifier_free_guidance_sampler_preserves_shape() -> None:
    model = ConditionalUNet(in_channels=3, num_conditions=4, base_channels=4, time_dim=8)
    scheduler = DDPMNoiseScheduler(timesteps=2)
    labels = torch.tensor([0, 2], dtype=torch.long)

    samples = sample_classifier_free_guidance(
        scheduler,
        model,
        (2, 3, 8, 8),
        labels,
        null_label=3,
        guidance_scale=1.5,
    )

    assert samples.shape == (2, 3, 8, 8)
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0


def test_classifier_guidance_sampler_preserves_shape() -> None:
    model = ConditionalUNet(in_channels=3, num_conditions=1, base_channels=4, time_dim=8)
    classifier = NoisyImageClassifier(
        in_channels=3,
        num_conditions=3,
        base_channels=4,
        time_dim=8,
    )
    scheduler = DDPMNoiseScheduler(timesteps=2)
    labels = torch.tensor([0, 2], dtype=torch.long)

    samples = sample_classifier_guidance(
        scheduler,
        model,
        classifier,
        (2, 3, 8, 8),
        labels,
        guidance_scale=1.0,
    )

    assert samples.shape == (2, 3, 8, 8)
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0


@pytest.mark.parametrize("guidance_mode", ["classifier-free", "classifier"])
def test_guided_training_smoke_run(tmp_path: Path, guidance_mode: str) -> None:
    image_root, json_root = _write_guidance_dataset(tmp_path)
    output_dir = tmp_path / f"guided_{guidance_mode}"
    config = GuidedConditionalDiffusionTrainConfig(
        data_root=image_root,
        json_root=json_root,
        output_dir=output_dir,
        image_size=8,
        fit_mode="resize",
        label_source="final-hash",
        max_images=2,
        batch_size=1,
        train_steps=1,
        timesteps=2,
        base_channels=4,
        time_dim=8,
        classifier_base_channels=4,
        guidance_mode=guidance_mode,
        guidance_scale=1.0,
        condition_dropout=0.5,
        sample_count=1,
        device="cpu",
    )

    result = train_guided_conditional_diffusion(config)

    assert result["guidance_mode"] == guidance_mode
    assert result["dataset_size"] == 2
    assert result["num_conditions"] == 2
    assert Path(result["checkpoint"]).exists()
    assert Path(result["sample_grid"]).exists()
    train_config = json.loads((output_dir / "train_config.json").read_text(encoding="utf-8"))
    assert train_config["guidance_mode"] == guidance_mode
    assert train_config["label_source"] == "final-hash"
    condition_to_idx = json.loads((output_dir / "condition_to_idx.json").read_text())
    assert set(condition_to_idx) == {"0xfinal-0", "0xfinal-1"}
