import torch
import numpy as np

def make_cosine_schedule(T=1000):
    """
    Cosine noise schedule from "Improved DDPM" (Nichol & Dhariwal 2021).
    Returns betas, alphas, alphas_cumprod as 1D torch tensors of length T.
    """
    s = 0.008   # small offset to prevent beta being too small near t=0

    # Step 1: build a tensor of timesteps from 0 to T (inclusive → T+1 values)
    steps = torch.arange(T + 1)                        # hint: torch.arange(T + 1)

    # Step 2: compute alphas_cumprod directly from the cosine formula
    # ᾱ_t = cos((t/T + s) / (1 + s) * π/2)²
    alphas_cumprod = torch.cos((steps / T + s) / (1 + s) * torch.pi / 2) ** 2                # hint: torch.cos(...)  ** 2

    # Step 3: normalize so ᾱ_0 = 1.0 exactly
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Step 4: derive betas from alphas_cumprod
    # beta_t = 1 - (ᾱ_t / ᾱ_{t-1})
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Step 5: clamp betas to prevent instability
    betas = torch.clamp(betas, 1e-4, 0.999)  # hint: min=1e-4, max=0.999

    # Step 6: derive alphas from betas
    alphas = 1 - betas                         # hint: 1 - betas

    # Step 7: recompute alphas_cumprod from the clamped betas
    # (now length T instead of T+1)
    alphas_cumprod = torch.cumprod(alphas, dim=0)                # hint: torch.cumprod(alphas, dim=0)

    return betas, alphas, alphas_cumprod


def forward_diffusion(x_0, t, alphas_cumprod):
    """
    Adds noise to clean trajectory x_0 at timestep t.
    Closed form: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * noise

    Args:
        x_0:            (B, 43, 6) clean future trajectory
        t:              (B,) integer timesteps
        alphas_cumprod: (T,) precomputed schedule

    Returns:
        x_t:   (B, 43, 6) noisy trajectory
        noise: (B, 43, 6) the noise that was added (this is what the model learns to predict)
    """
    # Step 1: sample random gaussian noise, same shape as x_0
    noise = torch.randn_like(x_0)                         # hint: torch.randn_like(x_0)

    # Step 2: gather ᾱ_t for each sample in the batch
    # alphas_cumprod[t] gives one scalar per sample → reshape to (B, 1, 1) to broadcast
    a_bar = alphas_cumprod[t].view(-1, 1, 1)  # hint: (-1, 1, 1)

    # Step 3: apply the closed form
    x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1-a_bar) * noise   # hint: sqrt(a_bar) and sqrt(1-a_bar)

    return x_t, noise


betas, alphas, alphas_cumprod = make_cosine_schedule(T=1000)
#print(betas.shape)          # torch.Size([1000])
#print(alphas_cumprod[0])    # should be close to 1.0
#print(alphas_cumprod[-1])   # should be close to 0.0