import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image

import diffusion_hash_inv.models.conditional_diffusion as conditional_diffusion_module
from diffusion_hash_inv.models.conditional_diffusion import (
    ConditionalDiffusionTrainConfig,
    ConditionalUNet,
    DDPMNoiseScheduler,
    GeneratedImageDataset,
    _ensure_square_batch,
    cleanup_torch_resources,
    build_beta_schedule,
    discover_generated_image_samples,
    discover_loop_image_samples,
    resolve_train_steps,
    save_image_grid,
    trace_timesteps,
    train_conditional_diffusion,
)


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def test_generated_image_dataset_uses_message_png_and_final_hash_labels(tmp_path: Path) -> None:
    root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    _write_png(root / "RUN_0001" / "message.png", (28, 28), (255, 0, 0, 255))
    _write_png(
        root / "RUN_0001" / "4th Step" / "1st Round" / "57th Loop.png",
        (112, 112),
        (0, 255, 0, 255),
    )
    _write_png(root / "RUN_0002" / "message.png", (896, 28), (0, 0, 255, 255))
    _write_png(root / "RUN_0002" / "1st Step.png", (16, 16), (4, 5, 6, 255))

    for run_id, final_hash in [("RUN_0001", "0xfinal-run1"), ("RUN_0002", "0xfinal-run2")]:
        json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "Message": {"Hex": "0xmessage"},
                    "Logs": {"4th Step": "0xlegacy-step4"},
                    "Generated hash": final_hash,
                    "Correct   hash": final_hash,
                }
            ),
            encoding="utf-8",
        )

    samples, condition_to_idx = discover_generated_image_samples(
        root,
        json_root=json_root,
    )

    assert len(samples) == 2
    assert all(sample.path.name == "message.png" for sample in samples)
    assert set(condition_to_idx) == {"0xfinal-run1", "0xfinal-run2"}

    dataset = GeneratedImageDataset(
        root,
        json_root=json_root,
        image_size=32,
        channels=3,
        fit_mode="pad",
    )
    image, label, _loop_meta = dataset[0]
    with Image.open(dataset.samples[0].path) as source:
        expected_side = max(source.width, source.height)

    assert image.shape == (3, expected_side, expected_side)
    assert image.dtype == torch.float32
    assert label.dtype == torch.long
    assert image.min() >= -1.0
    assert image.max() <= 1.0
    assert dataset.num_conditions == 2


def test_generated_image_dataset_explicitly_accepts_final_hash_labels(tmp_path: Path) -> None:
    root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    _write_png(root / "RUN_0001" / "message.png", (28, 28), (255, 0, 0, 255))
    _write_png(root / "RUN_0002" / "message.png", (28, 28), (0, 0, 255, 255))

    for run_id, step4_value, final_hash in [
        ("RUN_0001", "0xstep4-run1", "0xfinal-run1"),
        ("RUN_0002", "0xstep4-run2", "0xfinal-run2"),
    ]:
        json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "Message": {"Hex": "0xmessage"},
                    "Logs": {"4th Step": step4_value},
                    "Generated hash": final_hash,
                    "Correct   hash": final_hash,
                }
            ),
            encoding="utf-8",
        )

    samples, condition_to_idx = discover_generated_image_samples(
        root,
        json_root=json_root,
        label_source="final-hash",
    )

    assert len(samples) == 2
    assert set(condition_to_idx) == {"0xfinal-run1", "0xfinal-run2"}
    assert {sample.condition for sample in samples} == {"0xfinal-run1", "0xfinal-run2"}

    dataset = GeneratedImageDataset(
        root,
        json_root=json_root,
        image_size=32,
        channels=3,
        fit_mode="pad",
        label_source="final-hash",
    )

    assert dataset.label_source == "final-hash"
    assert dataset.num_conditions == 2
    assert set(dataset.condition_to_idx) == {"0xfinal-run1", "0xfinal-run2"}


def test_generated_image_dataset_requires_final_hash_label_in_json(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "MD5_256_2026-05-09 14-13-27_0000"
    _write_png(image_root / run_id / "message.png", (16, 16), (1, 2, 3, 255))
    json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"Message": {"Hex": "0xmessage"}, "Logs": {"1st Step": "0xstep1"}}),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Generated hash"):
        GeneratedImageDataset(
            image_root,
            json_root=json_root,
            image_size=16,
        )


def test_generated_image_dataset_rejects_empty_json_payload_file(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "MD5_256_2026-05-09 14-13-27_0000"
    _write_png(image_root / run_id / "message.png", (16, 16), (1, 2, 3, 255))
    json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON payload file is empty"):
        GeneratedImageDataset(
            image_root,
            json_root=json_root,
            image_size=16,
        )


def test_generated_image_dataset_supports_equal_area_reshape_mode(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "RUN_0001"
    _write_png(image_root / run_id / "message.png", (8, 2), (1, 2, 3, 255))
    json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "Message": {"Hex": "0xmessage"},
                "Logs": {"4th Step": "0xstep4"},
                "Generated hash": "0xfinal",
            }
        ),
        encoding="utf-8",
    )

    dataset = GeneratedImageDataset(
        image_root,
        json_root=json_root,
        image_size=32,
        fit_mode="reshape",
    )
    image, _, _loop_meta = dataset[0]

    assert image.shape == (3, 4, 4)


def test_generated_image_dataset_supports_height_flatten_mode(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "RUN_0001"
    _write_png(image_root / run_id / "message.png", (112, 28), (1, 2, 3, 255))
    json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "Message": {"Hex": "0xmessage"},
                "Logs": {"4th Step": "0xstep4"},
                "Generated hash": "0xfinal",
            }
        ),
        encoding="utf-8",
    )

    dataset = GeneratedImageDataset(
        image_root,
        json_root=json_root,
        image_size=32,
        channels=3,
        fit_mode="height-flatten",
    )
    image, _, _loop_meta = dataset[0]

    assert dataset.channels == 3
    assert image.shape == (3, 56, 56)


def test_conditional_unet_and_scheduler_preserve_shape() -> None:
    model = ConditionalUNet(in_channels=3, num_conditions=4, base_channels=8, time_dim=16)
    scheduler = DDPMNoiseScheduler(timesteps=8)
    x0 = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 3], dtype=torch.long)
    timesteps = torch.tensor([0, 7], dtype=torch.long)
    noise = torch.randn_like(x0)

    noised = scheduler.q_sample(x0, timesteps, noise)
    pred_noise = model(noised, timesteps, labels)

    assert noised.shape == x0.shape
    assert pred_noise.shape == x0.shape


def test_ensure_square_batch_resizes_non_square_tensor() -> None:
    images = torch.randn(2, 3, 12, 20)

    squared = _ensure_square_batch(images)

    assert squared.shape == (2, 3, 20, 20)


def test_ddpm_noise_scheduler_accepts_custom_betas() -> None:
    betas = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float32)

    scheduler = DDPMNoiseScheduler(betas=betas)

    assert scheduler.timesteps == 3
    torch.testing.assert_close(scheduler.betas.cpu(), betas)


def test_build_beta_schedule_uses_file_beta_length(tmp_path: Path) -> None:
    beta_path = tmp_path / "betas.json"
    beta_path.write_text(json.dumps({"betas": [0.01, 0.03]}), encoding="utf-8")
    config = ConditionalDiffusionTrainConfig(
        beta_schedule="file",
        beta_values_path=beta_path,
        timesteps=4,
    )

    betas = build_beta_schedule(config)

    assert betas is not None
    assert betas.shape == (2,)
    assert betas[0] == 0.01
    assert betas[-1] == 0.03
    assert (betas > 0).all()
    assert (betas < 1).all()


def test_build_beta_schedule_rejects_empty_beta_json_file(tmp_path: Path) -> None:
    beta_path = tmp_path / "betas.json"
    beta_path.write_text("", encoding="utf-8")
    config = ConditionalDiffusionTrainConfig(
        beta_schedule="file",
        beta_values_path=beta_path,
        timesteps=4,
    )

    with pytest.raises(ValueError, match="Beta values file is empty"):
        build_beta_schedule(config)


def test_build_beta_schedule_rejects_invalid_beta_json_file(tmp_path: Path) -> None:
    beta_path = tmp_path / "betas.json"
    beta_path.write_text("{", encoding="utf-8")
    config = ConditionalDiffusionTrainConfig(
        beta_schedule="file",
        beta_values_path=beta_path,
        timesteps=4,
    )

    with pytest.raises(ValueError, match="Invalid JSON in beta values file"):
        build_beta_schedule(config)


def test_build_beta_schedule_linear_uses_timesteps_when_not_auto() -> None:
    config = ConditionalDiffusionTrainConfig(
        beta_schedule="linear",
        timesteps=7,
    )

    betas = build_beta_schedule(config)

    assert betas is None


def test_build_beta_schedule_linear_auto_uses_hash_approach_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyAnalyze:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def summarize_beta_schedules(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(mean=[1.0, 2.0, 3.0, 4.0])

    class DummyBetaScheduler:
        def __init__(self, beta_min: float, beta_max: float) -> None:
            self.beta_min = beta_min
            self.beta_max = beta_max

        def approach1(self, _mean: object) -> SimpleNamespace:
            return SimpleNamespace(rescaled_candidate=[0.11, 0.12, 0.13, 0.14])

        def approach2(self, _mean: object) -> SimpleNamespace:
            return SimpleNamespace(candidate=[0.21, 0.22, 0.23, 0.24])

    monkeypatch.setattr(conditional_diffusion_module, "Analyze", DummyAnalyze)
    monkeypatch.setattr(conditional_diffusion_module, "BetaScheduler", DummyBetaScheduler)
    config = ConditionalDiffusionTrainConfig(
        beta_schedule="linear",
        timesteps="auto",
        beta_start=0.001,
        beta_end=0.009,
    )

    betas = build_beta_schedule(config)

    assert betas is not None
    assert betas.shape == (4,)
    assert betas.tolist() == pytest.approx([0.001, 0.0036666667, 0.0063333333, 0.009])


def test_build_beta_schedule_linear_auto_rejects_hash_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyAnalyze:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def summarize_beta_schedules(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(mean=[1.0, 2.0, 3.0])

    class DummyBetaScheduler:
        def __init__(self, beta_min: float, beta_max: float) -> None:
            self.beta_min = beta_min
            self.beta_max = beta_max

        def approach1(self, _mean: object) -> SimpleNamespace:
            return SimpleNamespace(rescaled_candidate=[0.11, 0.12, 0.13])

        def approach2(self, _mean: object) -> SimpleNamespace:
            return SimpleNamespace(candidate=[0.21, 0.22, 0.23, 0.24])

    monkeypatch.setattr(conditional_diffusion_module, "Analyze", DummyAnalyze)
    monkeypatch.setattr(conditional_diffusion_module, "BetaScheduler", DummyBetaScheduler)
    config = ConditionalDiffusionTrainConfig(beta_schedule="linear", timesteps="auto")

    with pytest.raises(ValueError, match="Hash approach schedule length mismatch"):
        build_beta_schedule(config)


def test_resolve_train_steps_uses_epochs_when_provided() -> None:
    assert resolve_train_steps(
        dataset_size=10,
        batch_size=4,
        train_steps=99,
        epochs=2,
    ) == 6


def test_resolve_train_steps_uses_train_steps_without_epochs() -> None:
    assert resolve_train_steps(
        dataset_size=10,
        batch_size=4,
        train_steps=99,
        epochs=None,
    ) == 99


def test_trace_timesteps_are_evenly_spaced_unique_values() -> None:
    assert trace_timesteps(timesteps=10, trace_steps=4) == [0, 3, 6, 9]
    assert trace_timesteps(timesteps=3, trace_steps=8) == [0, 1, 2]
    assert trace_timesteps(timesteps=3, trace_steps=0) == []


def test_save_image_grid_saves_individual_files_for_multiple_samples(tmp_path: Path) -> None:
    images = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    path = tmp_path / "grid.png"

    save_image_grid(images, labels, ["A", "B"], path)

    assert not path.exists()
    assert (tmp_path / "grid_000.png").exists()
    assert (tmp_path / "grid_001.png").exists()
    labels_payload = json.loads((tmp_path / "grid.labels.json").read_text(encoding="utf-8"))
    assert labels_payload[0]["file"] == "grid_000.png"
    assert labels_payload[1]["file"] == "grid_001.png"


def test_training_can_save_forward_and_reverse_process_traces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "MD5_256_2026-05-09 14-13-27_0000"
    _write_png(image_root / run_id / "message.png", (16, 16), (1, 2, 3, 255))
    _write_png(image_root / run_id / "3rd Step.png", (16, 16), (4, 5, 6, 255))
    json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "Message": {"Hex": "0xmessage"},
                "Logs": {
                    "4th Step": {"A": "0x01", "B": "0x02"},
                },
                "Generated hash": "0xfinal",
            }
        ),
        encoding="utf-8",
    )
    config = ConditionalDiffusionTrainConfig(
        data_root=image_root,
        json_root=json_root,
        output_dir=tmp_path / "train_output",
        image_size=8,
        batch_size=1,
        train_steps=1,
        timesteps=4,
        base_channels=4,
        time_dim=8,
        sample_count=1,
        save_process_traces=True,
        save_train_batches_every=1,
        trace_sample_count=1,
        trace_steps=2,
        device="cpu",
    )

    result = train_conditional_diffusion(config)
    captured = capsys.readouterr()

    trace_root = result["process_traces"]
    assert trace_root is not None
    trace_root = Path(trace_root)
    assert (trace_root / "forward" / "png" / "x0.png").exists()
    assert (trace_root / "forward" / "png" / "t_000000.png").exists()
    assert (trace_root / "forward" / "png" / "t_000001.png").exists()
    assert (trace_root / "forward" / "png" / "t_000002.png").exists()
    assert (trace_root / "forward" / "png" / "t_000003.png").exists()
    assert (trace_root / "forward" / "json" / "x0.labels.json").exists()
    assert (trace_root / "reverse" / "png" / "xT_noise.png").exists()
    assert (trace_root / "reverse" / "png" / "t_000003.png").exists()
    assert (trace_root / "reverse" / "png" / "t_000002.png").exists()
    assert (trace_root / "reverse" / "png" / "t_000001.png").exists()
    assert (trace_root / "reverse" / "png" / "t_000000.png").exists()
    assert (trace_root / "reverse" / "json" / "xT_noise.labels.json").exists()
    assert "[reshape] mode=reshape source=16x16 output=16x16 channels=3" in captured.out
    assert "[forward-trace] saving x0 + 4 noising steps" in captured.out
    assert "[forward-trace] step=000000" in captured.out
    assert "[forward-trace] step=000003" in captured.out
    assert "[reverse-trace] saving xT + 4 denoising steps" in captured.out
    assert "[reverse-trace] step=000003" in captured.out
    assert "[reverse-trace] step=000000" in captured.out
    assert Path(result["sample_grid"]).name == "final.png"
    assert Path(result["sample_source_grid"]).name == "final.source.png"
    assert Path(result["sample_with_source_grid"]).name == "final.with_source.png"
    assert Path(result["sample_source_grid"]).exists()
    assert not Path(result["sample_with_source_grid"]).exists()
    assert Path(result["sample_with_source_grid"]).with_name("final.with_source.source.png").exists()
    assert Path(result["sample_with_source_grid"]).with_name("final.with_source.generated.png").exists()
    assert (Path(result["train_batches"]) / "step_000001.png").exists()
    batch_metadata = Path(result["train_batches"]) / "step_000001.batch.json"
    assert batch_metadata.exists()
    payload = json.loads(batch_metadata.read_text(encoding="utf-8"))
    assert payload["step"] == 1
    assert payload["samples"]


def test_cleanup_torch_resources_clears_state_dict() -> None:
    state = {"tensor": torch.ones(1), "text": "keep"}

    cleanup_torch_resources(state)

    assert state == {}


# ---------------------------------------------------------------------------
# Temporal conditioning tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "temporal_conditioning",
    ["loop-sinusoidal", "loop-structured", "loop-sequence"],
)
def test_conditional_unet_temporal_modes_preserve_shape(temporal_conditioning: str) -> None:
    model = ConditionalUNet(
        in_channels=3,
        num_conditions=4,
        base_channels=8,
        time_dim=16,
        temporal_conditioning=temporal_conditioning,  # type: ignore[arg-type]
        max_loop_count=8,
    )
    x = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 3], dtype=torch.long)
    timesteps = torch.tensor([0, 7], dtype=torch.long)
    # loop_meta: [loop_idx, loop_count, loop_start, loop_end]
    loop_meta = torch.tensor([[0.0, 8.0, 0.0, 0.125], [7.0, 8.0, 0.875, 1.0]])

    out = model(x, timesteps, labels, loop_meta)

    assert out.shape == x.shape


def test_conditional_unet_temporal_class_without_loop_meta_preserves_shape() -> None:
    """'class' mode must work with loop_meta=None for backward compat."""
    model = ConditionalUNet(in_channels=3, num_conditions=4, base_channels=8, time_dim=16)
    x = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 3], dtype=torch.long)
    timesteps = torch.tensor([0, 7], dtype=torch.long)

    out = model(x, timesteps, labels)

    assert out.shape == x.shape


def test_discover_loop_image_samples_finds_loop_pngs(tmp_path: Path) -> None:
    root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "RUN_0001"

    # Write 3 loop images (ordinal 1st/2nd/3rd = idx 0/1/2)
    for loop_label in ("1st Loop", "2nd Loop", "3rd Loop"):
        _write_png(
            root / run_id / "4th Step" / "1st Round" / f"{loop_label}.png",
            (28, 28),
            (255, 0, 0, 255),
        )

    json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "Message": {"Hex": "0xmsg"},
                "Logs": {"4th Step": "0xstep4"},
                "Generated hash": "0xfinal",
            }
        ),
        encoding="utf-8",
    )

    samples, condition_to_idx = discover_loop_image_samples(root, json_root=json_root)

    assert len(samples) == 3
    assert all(hasattr(s, "loop_idx") for s in samples)
    loop_indices = sorted(s.loop_idx for s in samples)
    assert loop_indices == [0, 1, 2]
    assert len(condition_to_idx) > 0


def test_dataset_loop_meta_tensor_has_correct_layout(tmp_path: Path) -> None:
    root = tmp_path / "images"
    json_root = tmp_path / "output" / "json"
    run_id = "RUN_0001"
    _write_png(root / run_id / "message.png", (16, 16), (1, 2, 3, 255))
    json_path = json_root / "2026-05-09 14-13-27" / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "Message": {"Hex": "0xmsg"},
                "Logs": {"4th Step": "0xstep4"},
                "Generated hash": "0xfinal",
            }
        ),
        encoding="utf-8",
    )

    dataset = GeneratedImageDataset(root, json_root=json_root, image_size=16)
    _image, _label, loop_meta = dataset[0]

    assert loop_meta.shape == (4,)
    assert loop_meta.dtype == torch.float32
    # Default sample has loop_idx=0, loop_count=1 → start=0, end=1
    assert loop_meta[0].item() == pytest.approx(0.0)  # loop_idx
    assert loop_meta[1].item() == pytest.approx(1.0)  # loop_count
