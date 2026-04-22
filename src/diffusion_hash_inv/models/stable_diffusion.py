"""
Minimal Stable Diffusion-style latent diffusion model.

The implementation is intentionally lightweight and self-contained. It mirrors
the main subsystems of Stable Diffusion without targeting checkpoint
compatibility:

- convolutional autoencoder for image <-> latent conversion
- token-based text conditioner
- cross-attention latent denoiser
- classifier-free guidance sampler

Images are expected to be normalized to ``[-1, 1]``. Prompts are represented as
integer token tensors of shape ``[batch, sequence_length]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_hash_inv.models.two_image_fusion import SinusoidalTimeEmbedding


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _resolve_num_heads(channels: int, preferred: int) -> int:
    for heads in range(min(channels, preferred), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class LatentResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + residual


class CrossAttention2d(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = _group_norm(channels)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=_resolve_num_heads(channels, num_heads),
            batch_first=True,
        )
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query = self.norm(x).flatten(2).transpose(1, 2)
        key_value = self.context_proj(self.context_norm(context))
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.output(attended)


class LatentAutoencoder(nn.Module):
    """
    Small convolutional autoencoder with a fixed 4x spatial compression ratio.
    """

    downsample_factor = 4

    def __init__(
        self,
        image_channels: int = 3,
        latent_channels: int = 4,
        hidden_channels: int = 32,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        mid_channels = hidden_channels * 2

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder_input = nn.Conv2d(latent_channels, mid_channels, kernel_size=3, padding=1)
        self.decoder_block1 = nn.Conv2d(mid_channels, hidden_channels, kernel_size=3, padding=1)
        self.decoder_block2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.decoder_output = nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1)

    def _validate_spatial_size(self, tensor: torch.Tensor) -> None:
        height, width = tensor.shape[-2:]
        factor = self.downsample_factor
        if height % factor != 0 or width % factor != 0:
            raise ValueError(f"image height and width must be divisible by {factor}")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self._validate_spatial_size(images)
        return self.encoder(images) * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(latents / self.scaling_factor)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.silu(self.decoder_block1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.silu(self.decoder_block2(x))
        return torch.tanh(self.decoder_output(x))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


class TextConditionEncoder(nn.Module):
    """
    Transformer-based text conditioner operating on integer prompt tokens.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        embed_dim: int = 128,
        max_length: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.null_token_id = 0
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=_resolve_num_heads(embed_dim, num_heads),
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)

    def null_prompt(self, batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size, sequence_length),
            self.null_token_id,
            device=device,
            dtype=torch.long,
        )

    def forward(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must have shape [batch, sequence_length]")
        if prompt_tokens.shape[1] > self.max_length:
            raise ValueError(f"prompt length must be <= {self.max_length}")

        embeddings = self.token_embedding(prompt_tokens)
        embeddings = embeddings + self.position_embedding[:, : prompt_tokens.shape[1]]
        return self.final_norm(self.encoder(embeddings))


class LatentDiffusionUNet(nn.Module):
    """
    Small UNet-like denoiser operating in latent space with cross-attention.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        time_dim: int = 256,
        context_dim: int = 128,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.down_block = LatentResidualBlock(base_channels, base_channels, time_dim)
        self.downsample = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.mid_input = LatentResidualBlock(base_channels, base_channels * 2, time_dim)
        self.mid_block = LatentResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.mid_attention = CrossAttention2d(
            channels=base_channels * 2,
            context_dim=context_dim,
            num_heads=attention_heads,
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
        )
        self.up_block = LatentResidualBlock(base_channels * 3, base_channels, time_dim)
        self.up_attention = CrossAttention2d(
            channels=base_channels,
            context_dim=context_dim,
            num_heads=attention_heads,
        )
        self.output = nn.Sequential(
            _group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latents.shape[0] != timesteps.shape[0] or noisy_latents.shape[0] != context.shape[0]:
            raise ValueError("batch size for latents, timesteps, and context must match")

        time_embedding = self.time_embedding(timesteps)
        x = self.input(noisy_latents)
        skip = self.down_block(x, time_embedding)
        x = self.downsample(skip)
        x = self.mid_input(x, time_embedding)
        x = self.mid_block(x, time_embedding)
        x = self.mid_attention(x, context)
        x = self.upsample(x)
        x = torch.cat((skip, x), dim=1)
        x = self.up_block(x, time_embedding)
        x = self.up_attention(x, context)
        return self.output(x)


class StableDiffusion(nn.Module):
    """
    Minimal latent diffusion model with text conditioning and guided sampling.
    """

    def __init__(
        self,
        image_channels: int = 3,
        latent_channels: int = 4,
        autoencoder_channels: int = 32,
        unet_channels: int = 64,
        time_dim: int = 256,
        text_embed_dim: int = 128,
        vocab_size: int = 2048,
        max_prompt_length: int = 32,
        num_train_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        prompt_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.prompt_dropout = prompt_dropout
        self.autoencoder = LatentAutoencoder(
            image_channels=image_channels,
            latent_channels=latent_channels,
            hidden_channels=autoencoder_channels,
        )
        self.text_encoder = TextConditionEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            max_length=max_prompt_length,
        )
        self.unet = LatentDiffusionUNet(
            latent_channels=latent_channels,
            base_channels=unet_channels,
            time_dim=time_dim,
            context_dim=text_embed_dim,
        )

        betas = torch.linspace(beta_start, beta_end, num_train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        extracted = values.gather(0, timesteps)
        return extracted.view(shape[0], 1, 1, 1)

    def _sampling_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        if steps < 1 or steps > self.num_train_steps:
            raise ValueError(
                f"num_inference_steps must be between 1 and {self.num_train_steps}"
            )
        timesteps = torch.linspace(
            self.num_train_steps - 1,
            0,
            steps=steps,
            device=device,
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        if timesteps[-1].item() != 0:
            timesteps = torch.cat((timesteps, torch.zeros(1, device=device, dtype=torch.long)))
        return timesteps

    def _drop_prompt_tokens(
        self,
        prompt_tokens: torch.Tensor,
        dropout: float | None,
    ) -> torch.Tensor:
        effective_dropout = self.prompt_dropout if dropout is None else dropout
        if effective_dropout <= 0.0:
            return prompt_tokens

        batch = prompt_tokens.shape[0]
        drop_mask = torch.rand(batch, 1, device=prompt_tokens.device) < effective_dropout
        null_tokens = torch.full_like(prompt_tokens, self.text_encoder.null_token_id)
        return torch.where(drop_mask, null_tokens, prompt_tokens)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(images)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(latents)

    def encode_prompts(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(prompt_tokens)

    def q_sample(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(latents)

        alpha_bar_t = self._extract(self.alpha_bars, timesteps, latents.shape)
        return alpha_bar_t.sqrt() * latents + (1.0 - alpha_bar_t).sqrt() * noise

    def training_loss(
        self,
        images: torch.Tensor,
        prompt_tokens: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        prompt_dropout: float | None = None,
    ) -> torch.Tensor:
        if images.shape[0] != prompt_tokens.shape[0]:
            raise ValueError("images and prompt_tokens must have the same batch size")

        latents = self.encode_image(images)
        batch_size = latents.shape[0]
        device = latents.device

        if timesteps is None:
            timesteps = torch.randint(0, self.num_train_steps, (batch_size,), device=device)
        if noise is None:
            noise = torch.randn_like(latents)

        conditioned_tokens = self._drop_prompt_tokens(prompt_tokens, prompt_dropout)
        noisy_latents = self.q_sample(latents, timesteps, noise)
        context = self.encode_prompts(conditioned_tokens)
        predicted_noise = self.unet(noisy_latents, timesteps, context)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        prompt_tokens: torch.Tensor,
        image_size: tuple[int, int],
        guidance_scale: float = 7.5,
        num_inference_steps: int | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must have shape [batch, sequence_length]")

        height, width = image_size
        factor = self.autoencoder.downsample_factor
        if height % factor != 0 or width % factor != 0:
            raise ValueError(f"image height and width must be divisible by {factor}")

        batch_size = prompt_tokens.shape[0]
        device = prompt_tokens.device
        latent_height = height // factor
        latent_width = width // factor
        sequence_length = prompt_tokens.shape[1]
        steps = num_inference_steps or self.num_train_steps
        sampling_timesteps = self._sampling_timesteps(steps, device)

        if latents is None:
            current = torch.randn(
                batch_size,
                self.autoencoder.latent_channels,
                latent_height,
                latent_width,
                device=device,
            )
        else:
            expected_shape = (
                batch_size,
                self.autoencoder.latent_channels,
                latent_height,
                latent_width,
            )
            if latents.shape != expected_shape:
                raise ValueError(f"latents must have shape {expected_shape}")
            current = latents

        conditional_context = self.encode_prompts(prompt_tokens)
        unconditional_context = None
        if guidance_scale != 1.0:
            null_prompt = self.text_encoder.null_prompt(batch_size, sequence_length, device)
            unconditional_context = self.encode_prompts(null_prompt)

        for index, timestep in enumerate(sampling_timesteps):
            timestep_batch = torch.full(
                (batch_size,),
                timestep.item(),
                device=device,
                dtype=torch.long,
            )
            predicted_noise = self.unet(current, timestep_batch, conditional_context)
            if unconditional_context is not None:
                unconditional_noise = self.unet(current, timestep_batch, unconditional_context)
                predicted_noise = unconditional_noise + guidance_scale * (
                    predicted_noise - unconditional_noise
                )

            alpha_bar_t = self._extract(self.alpha_bars, timestep_batch, current.shape)
            predicted_clean = (
                current - (1.0 - alpha_bar_t).sqrt() * predicted_noise
            ) / alpha_bar_t.sqrt()

            if index == len(sampling_timesteps) - 1:
                current = predicted_clean
                continue

            next_timestep = sampling_timesteps[index + 1]
            next_batch = torch.full(
                (batch_size,),
                next_timestep.item(),
                device=device,
                dtype=torch.long,
            )
            alpha_bar_next = self._extract(self.alpha_bars, next_batch, current.shape)
            current = alpha_bar_next.sqrt() * predicted_clean + (
                1.0 - alpha_bar_next
            ).sqrt() * predicted_noise

        return self.decode_latents(current)
