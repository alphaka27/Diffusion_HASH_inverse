import pytest
import torch

from diffusion_hash_inv.models import LatentAutoencoder, StableDiffusion, TextConditionEncoder


def test_latent_autoencoder_round_trip_preserves_expected_shapes():
    autoencoder = LatentAutoencoder(
        image_channels=3,
        latent_channels=4,
        hidden_channels=16,
    )
    images = torch.randn(2, 3, 16, 16).tanh()

    latents = autoencoder.encode(images)
    reconstructed = autoencoder.decode(latents)

    assert latents.shape == (2, 4, 4, 4)
    assert reconstructed.shape == images.shape
    assert torch.isfinite(reconstructed).all()
    assert reconstructed.max() <= 1.0
    assert reconstructed.min() >= -1.0


def test_text_condition_encoder_rejects_prompt_lengths_beyond_max():
    encoder = TextConditionEncoder(vocab_size=64, embed_dim=32, max_length=4)
    prompt_tokens = torch.randint(0, 64, (2, 5))

    with pytest.raises(ValueError, match="prompt length"):
        encoder(prompt_tokens)


def test_stable_diffusion_training_loss_is_scalar():
    model = StableDiffusion(
        image_channels=3,
        latent_channels=4,
        autoencoder_channels=16,
        unet_channels=16,
        time_dim=32,
        text_embed_dim=32,
        vocab_size=128,
        max_prompt_length=8,
        num_train_steps=8,
        prompt_dropout=0.0,
    )
    images = torch.randn(2, 3, 16, 16).tanh()
    prompt_tokens = torch.randint(1, 128, (2, 8))

    loss = model.training_loss(images, prompt_tokens)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_stable_diffusion_sampling_returns_image_tensor():
    model = StableDiffusion(
        image_channels=3,
        latent_channels=4,
        autoencoder_channels=16,
        unet_channels=16,
        time_dim=32,
        text_embed_dim=32,
        vocab_size=128,
        max_prompt_length=8,
        num_train_steps=8,
        prompt_dropout=0.0,
    )
    prompt_tokens = torch.randint(1, 128, (2, 8))

    sample = model.sample(
        prompt_tokens=prompt_tokens,
        image_size=(16, 16),
        guidance_scale=4.0,
        num_inference_steps=4,
    )

    assert sample.shape == (2, 3, 16, 16)
    assert torch.isfinite(sample).all()
    assert sample.max() <= 1.0
    assert sample.min() >= -1.0


def test_classifier_free_guidance_changes_output_with_same_initial_latents():
    model = StableDiffusion(
        image_channels=3,
        latent_channels=4,
        autoencoder_channels=16,
        unet_channels=16,
        time_dim=32,
        text_embed_dim=32,
        vocab_size=128,
        max_prompt_length=8,
        num_train_steps=8,
        prompt_dropout=0.0,
    )
    prompt_tokens = torch.randint(1, 128, (2, 8))
    initial_latents = torch.randn(2, 4, 4, 4)

    unguided = model.sample(
        prompt_tokens=prompt_tokens,
        image_size=(16, 16),
        guidance_scale=0.0,
        num_inference_steps=4,
        latents=initial_latents.clone(),
    )
    guided = model.sample(
        prompt_tokens=prompt_tokens,
        image_size=(16, 16),
        guidance_scale=5.0,
        num_inference_steps=4,
        latents=initial_latents.clone(),
    )

    assert not torch.allclose(unguided, guided)


def test_sampling_rejects_wrong_latent_shape():
    model = StableDiffusion(
        image_channels=3,
        latent_channels=4,
        autoencoder_channels=16,
        unet_channels=16,
        time_dim=32,
        text_embed_dim=32,
        vocab_size=128,
        max_prompt_length=8,
        num_train_steps=8,
        prompt_dropout=0.0,
    )
    prompt_tokens = torch.randint(1, 128, (2, 8))
    latents = torch.randn(2, 4, 8, 8)

    with pytest.raises(ValueError, match="latents must have shape"):
        model.sample(prompt_tokens=prompt_tokens, image_size=(16, 16), latents=latents)
