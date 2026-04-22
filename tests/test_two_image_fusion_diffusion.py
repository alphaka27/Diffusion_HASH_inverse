import torch
import pytest

from diffusion_hash_inv.models import FusionNoisePredictor, TwoImageFusionDiffusion
from diffusion_hash_inv.models.two_image_fusion import SinusoidalTimeEmbedding


def test_noise_predictor_preserves_target_shape():
    model = FusionNoisePredictor(image_channels=3, base_channels=16, time_dim=32)
    source_a = torch.randn(2, 3, 16, 16)
    source_b = torch.randn(2, 3, 16, 16)
    noisy_target = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([1, 3], dtype=torch.long)

    predicted_noise = model(source_a, source_b, noisy_target, timesteps)

    assert predicted_noise.shape == noisy_target.shape
    assert torch.isfinite(predicted_noise).all()


def test_noise_predictor_rejects_mismatched_shapes():
    model = FusionNoisePredictor(image_channels=3, base_channels=16, time_dim=32)
    source_a = torch.randn(2, 3, 16, 16)
    source_b = torch.randn(2, 3, 16, 16)
    noisy_target = torch.randn(2, 3, 8, 8)
    timesteps = torch.tensor([1, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="matching shapes"):
        model(source_a, source_b, noisy_target, timesteps)


def test_sinusoidal_time_embedding_supports_odd_dimensions():
    embedding = SinusoidalTimeEmbedding(embedding_dim=5)
    timesteps = torch.tensor([0, 1, 2], dtype=torch.long)

    encoded = embedding(timesteps)

    assert encoded.shape == (3, 5)
    assert torch.all(encoded[:, -1] == 0)


def test_training_loss_is_finite_scalar():
    model = TwoImageFusionDiffusion(
        image_channels=3,
        base_channels=16,
        time_dim=32,
        num_steps=8,
    )
    source_a = torch.randn(2, 3, 16, 16)
    source_b = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)

    loss = model.training_loss(source_a, source_b, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_q_sample_matches_closed_form_when_noise_is_provided():
    model = TwoImageFusionDiffusion(
        image_channels=3,
        base_channels=16,
        time_dim=32,
        num_steps=8,
    )
    target = torch.ones(2, 3, 4, 4)
    timesteps = torch.tensor([0, 3], dtype=torch.long)
    noise = torch.full_like(target, 0.25)

    noisy = model.q_sample(target, timesteps, noise)
    alpha_bar = model.alpha_bars.gather(0, timesteps).view(2, 1, 1, 1)
    expected = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise

    assert torch.allclose(noisy, expected)


def test_sampling_returns_clamped_image_tensor():
    model = TwoImageFusionDiffusion(
        image_channels=3,
        base_channels=16,
        time_dim=32,
        num_steps=4,
    )
    source_a = torch.randn(2, 3, 16, 16)
    source_b = torch.randn(2, 3, 16, 16)

    sample = model.sample(source_a, source_b)

    assert sample.shape == source_a.shape
    assert torch.isfinite(sample).all()
    assert sample.max() <= 1.0
    assert sample.min() >= -1.0


def test_sampling_rejects_mismatched_condition_shapes():
    model = TwoImageFusionDiffusion(
        image_channels=3,
        base_channels=16,
        time_dim=32,
        num_steps=4,
    )
    source_a = torch.randn(2, 3, 16, 16)
    source_b = torch.randn(2, 3, 8, 8)

    with pytest.raises(ValueError, match="matching shapes"):
        model.sample(source_a, source_b)
