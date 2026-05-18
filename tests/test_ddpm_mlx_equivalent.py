from __future__ import annotations

import pytest


def _load_module():
    try:
        import mlx.core as mx
        from diffusion_hash_inv.models import ddpm_mlx_equivalent as ddpm_mlx
    except Exception as exc:  # pragma: no cover - depends on local MLX/Metal setup.
        pytest.skip(f"MLX unavailable in this environment: {exc}")
    return mx, ddpm_mlx


def test_conditional_unet_shape_contracts() -> None:
    mx, ddpm_mlx = _load_module()
    model = ddpm_mlx.MLXConditionalUNet(
        in_channels=3,
        num_conditions=4,
        base_channels=4,
        time_dim=8,
    )
    x = mx.zeros((2, 4, 4, 3), dtype=mx.float32)
    timesteps = mx.array([0, 1], dtype=mx.int32)
    labels = mx.array([0, 3], dtype=mx.int32)

    assert model(x, timesteps, labels).shape == x.shape


def test_temporal_conditioning_shape_contracts() -> None:
    mx, ddpm_mlx = _load_module()
    x = mx.zeros((2, 4, 4, 3), dtype=mx.float32)
    timesteps = mx.array([0, 1], dtype=mx.int32)
    labels = mx.zeros((2,), dtype=mx.int32)
    loop_meta = mx.array(
        [
            [0.0, 64.0, 0.0, 0.015625],
            [1.0, 64.0, 0.015625, 0.03125],
        ],
        dtype=mx.float32,
    )
    for mode in ("loop-sinusoidal", "loop-structured", "loop-sequence"):
        model = ddpm_mlx.MLXConditionalUNet(
            in_channels=3,
            num_conditions=4,
            base_channels=4,
            time_dim=8,
            temporal_conditioning=mode,
            max_loop_count=64,
        )
        assert model(x, timesteps, labels, loop_meta).shape == x.shape


def test_unconditional_loop_classifier_and_scheduler_contracts() -> None:
    mx, ddpm_mlx = _load_module()
    x = mx.zeros((2, 4, 4, 3), dtype=mx.float32)
    timesteps = mx.array([0, 1], dtype=mx.int32)

    uncond = ddpm_mlx.MLXUnconditionalUNet(in_channels=3, base_channels=4, time_dim=8)
    assert uncond(x, timesteps).shape == x.shape

    loop = ddpm_mlx.MLXLoopConditionedUNet(
        in_channels=3,
        condition_shape=(4, 4),
        base_channels=4,
        time_dim=8,
        diffusion_timesteps=16,
    )
    conditions = mx.zeros((2, 4, 4), dtype=mx.float32)
    assert loop(x, timesteps, conditions).shape == x.shape

    classifier = ddpm_mlx.MLXNoisyImageClassifier(
        in_channels=3,
        num_conditions=5,
        base_channels=4,
        time_dim=8,
    )
    assert classifier(x, timesteps).shape == (2, 5)

    scheduler = ddpm_mlx.MLXImageDDPMScheduler(timesteps=4)
    noisy = scheduler.q_sample(x, timesteps)
    assert noisy.shape == x.shape
    assert ddpm_mlx.unconditional_diffusion_loss(uncond, scheduler, x).shape == ()


def test_mlx_train_step_helpers_return_scalar_losses() -> None:
    mx, ddpm_mlx = _load_module()
    try:
        import mlx.optimizers as optim
    except Exception as exc:  # pragma: no cover - depends on local MLX setup.
        pytest.skip(f"MLX optimizers unavailable in this environment: {exc}")

    x = mx.zeros((2, 4, 4, 3), dtype=mx.float32)
    labels = mx.array([0, 1], dtype=mx.int32)
    scheduler = ddpm_mlx.MLXImageDDPMScheduler(timesteps=4)

    conditional = ddpm_mlx.MLXConditionalUNet(
        in_channels=3,
        num_conditions=3,
        base_channels=4,
        time_dim=8,
    )
    conditional_step = ddpm_mlx.make_conditional_train_step(
        conditional,
        scheduler,
        optim.Adam(learning_rate=1e-3),
    )
    assert conditional_step(x, labels).shape == ()
    dropped = ddpm_mlx.apply_condition_dropout(labels, null_label=2, dropout=1.0)
    assert mx.all(dropped == 2).item()

    cfg_step = ddpm_mlx.make_classifier_free_train_step(
        conditional,
        scheduler,
        optim.Adam(learning_rate=1e-3),
        null_label=2,
        condition_dropout=0.1,
    )
    assert cfg_step(x, labels).shape == ()

    uncond = ddpm_mlx.MLXUnconditionalUNet(in_channels=3, base_channels=4, time_dim=8)
    uncond_step = ddpm_mlx.make_unconditional_train_step(
        uncond,
        scheduler,
        optim.Adam(learning_rate=1e-3),
    )
    assert uncond_step(x).shape == ()
