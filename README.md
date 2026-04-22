# Diffusion HASH inverse
Finding the hash inverse using a diffusion model

# SHA-256
## SHA-256 Properties
Message Size (bits): less than $2^{64}$  
Block Size (bits): $512 = 2^9$  
Word Size (bits): $32  = 2^5$
Message Digest Size (bits): 256  

# Two-Image Fusion Diffusion
The repository now includes a minimal conditional DDPM module for learning
triples of `(source_a, source_b, target)` image tensors.

Example:

```python
import torch

from diffusion_hash_inv.models import TwoImageFusionDiffusion

model = TwoImageFusionDiffusion(image_channels=3, num_steps=100)
source_a = torch.randn(4, 3, 64, 64)
source_b = torch.randn(4, 3, 64, 64)
target = torch.randn(4, 3, 64, 64)

loss = model.training_loss(source_a, source_b, target)
loss.backward()

with torch.no_grad():
    fused = model.sample(source_a, source_b)
```

# Stable Diffusion
The repository also includes a lightweight Stable Diffusion-style latent
diffusion module with:

- a convolutional autoencoder for image latents
- a transformer text conditioner
- a cross-attention latent UNet
- classifier-free guidance during sampling

Example:

```python
import torch

from diffusion_hash_inv.models import StableDiffusion

model = StableDiffusion(
    image_channels=3,
    latent_channels=4,
    text_embed_dim=64,
    max_prompt_length=16,
    num_train_steps=50,
)
images = torch.randn(2, 3, 32, 32).tanh()
prompt_tokens = torch.randint(1, 512, (2, 16))

loss = model.training_loss(images, prompt_tokens)
loss.backward()

with torch.no_grad():
    samples = model.sample(
        prompt_tokens=prompt_tokens,
        image_size=(32, 32),
        guidance_scale=7.5,
        num_inference_steps=10,
    )
```
