from pathlib import Path

import torch
from PIL import Image

from diffusion_hash_inv.models import TwoImageFusionDiffusion


FIXTURE_DIR = Path("data/images/MD5_128_2026-02-23 17-01-46_01")
TARGET_SIZE = (64, 64)


def _load_rgb_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize(size, Image.Resampling.BILINEAR)
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(size[1], size[0], 3).permute(2, 0, 1) / 255.0
    return tensor * 2.0 - 1.0


def test_md5_message_and_loop_images_can_drive_single_sample_fusion_training():
    torch.manual_seed(0)

    message = _load_rgb_tensor(FIXTURE_DIR / "message.png", TARGET_SIZE).unsqueeze(0)
    loop10 = _load_rgb_tensor(
        FIXTURE_DIR / "4th Step" / "1st Round" / "10th Loop.png",
        TARGET_SIZE,
    ).unsqueeze(0)
    loop11 = _load_rgb_tensor(
        FIXTURE_DIR / "4th Step" / "1st Round" / "11th Loop.png",
        TARGET_SIZE,
    ).unsqueeze(0)

    model = TwoImageFusionDiffusion(
        image_channels=3,
        base_channels=8,
        time_dim=16,
        num_steps=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    timesteps = torch.tensor([3], dtype=torch.long)
    noise = torch.randn_like(loop11)

    with torch.no_grad():
        initial_loss = model.training_loss(
            source_a=loop10,
            source_b=message,
            target=loop11,
            timesteps=timesteps,
            noise=noise,
        ).item()

    for _ in range(20):
        optimizer.zero_grad()
        loss = model.training_loss(
            source_a=loop10,
            source_b=message,
            target=loop11,
            timesteps=timesteps,
            noise=noise,
        )
        loss.backward()
        optimizer.step()

    final_loss = model.training_loss(
        source_a=loop10,
        source_b=message,
        target=loop11,
        timesteps=timesteps,
        noise=noise,
    ).item()
    sample = model.sample(loop10, message)

    assert message.shape == loop10.shape == loop11.shape == (1, 3, 64, 64)
    assert final_loss < initial_loss
    assert sample.shape == loop10.shape
    assert torch.isfinite(sample).all()
