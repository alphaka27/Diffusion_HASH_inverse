import torch

import torch.nn as nn
import torch.optim as optim

# Simple 1D diffusion model (denoising autoencoder style)
class SimpleDiffusionModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x, t):
        # t: timestep, can be used for conditioning (not used here for simplicity)
        return self.net(x)

# Forward diffusion: add noise
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    return x_start + noise * (0.1 + 0.9 * t / 100)

# Training loop example
def train_diffusion():
    dim = 16
    model = SimpleDiffusionModel(dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1000
    batch_size = 32

    for epoch in range(epochs):
        x_start = torch.randn(batch_size, dim)
        t = torch.randint(0, 100, (batch_size, 1), dtype=torch.float32)
        noise = torch.randn_like(x_start)
        x_noisy = q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t)
        loss = nn.MSELoss()(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_diffusion()
