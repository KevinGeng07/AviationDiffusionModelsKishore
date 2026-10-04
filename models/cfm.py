import numpy as np
import torch


def sample_flow_time(B, device = "cuda", t_mean=1.0, t_std=1.0):
    """
    Sample flow time t ~ logit-normal(t_mean, t_std) → sigmoid → (0, 1)
    This is the SD3 trick — concentrates training near t≈0.73
    instead of uniform sampling across all t.

    Args:
        B: batch size
    Returns:
        t: (B,) float tensor of flow times in (0, 1)
    """
    # Step 1: sample from normal distribution
    u = torch.randn(B, device=device) * t_std + t_mean                          # hint: torch.randn(B) * t_std + t_mean

    # Step 2: push through sigmoid to get t ∈ (0, 1)
    t = torch.sigmoid(u)                       # hint: torch.sigmoid(u)

    return t


def forward_cfm(x_0, t):
    """
    CFM forward process — straight line interpolation from noise to data.

    x_0: (B, 43, 6) clean future trajectory
    t:   (B,) flow time in (0, 1)

    Returns:
        x_t:   (B, 43, 6) interpolated trajectory
        noise: (B, 43, 6) the noise we started from
        v_tgt: (B, 43, 6) target velocity the model should predict
    """
    # Step 1: sample pure gaussian noise, same shape as x_0
    noise = torch.randn_like(x_0)                      # hint: torch.randn_like(x_0)

    # Step 2: reshape t for broadcasting across (B, 43, 6)
    t_broadcast = t.view(-1, 1, 1)        # hint: (-1, 1, 1)

    # Step 3: straight line interpolation
    # x_t = (1 - t) * noise + t * data
    x_t = (1-t_broadcast) * noise + t_broadcast * x_0   # hint: (1-t) and t

    # Step 4: target velocity is constant along the straight path
    # v = data - noise
    v_tgt = x_0 - noise                      # hint: x_0 - noise

    return x_t, noise, v_tgt


def euler_sample(model, obs, t_rel, n_samples=20, n_steps=20, device="cuda"):
    """
    CFM inference — simple Euler ODE integration from noise to data.
    Much simpler than DDIM: just follow the velocity field step by step.

    obs:       (B, 43, 6) normalized observed context
    t_rel:     (B, 86)    normalized relative timestamps
    n_samples: how many different futures to sample
    n_steps:   how many Euler steps (20 is enough for CFM)

    Returns: (n_samples, B, 43, 6) predicted futures
    """
    B  = obs.shape[0]
    dt = 1.0 / n_steps                         # hint: 1.0 / n_steps

    # Repeat obs and t_rel for all samples (same trick as DDIM)
    obs_rep  = obs.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, 43, 6)
    trel_rep = t_rel.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * B, 86)

    # Start from pure noise
    x = torch.randn(n_samples * B, 43, 6, device=device)                          # hint: torch.randn(n_samples * B, 43, 6, device=device)

    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            # Current flow time for this step
            t_val    = step * dt
            t_tensor = torch.full(
                (n_samples * B,), t_val, device=device, dtype=torch.float32
            )

            # Predict velocity field at current position and time
            # NOTE: CFM model takes t directly as float in (0,1)
            # not t/T like DDIM — t is already in (0,1)
            v_pred = model(obs_rep, x, t_tensor, trel_rep)
            # hint: (obs_rep, x, t_tensor, trel_rep)

            # Euler step: move in direction of predicted velocity
            x = x + v_pred * dt
            # hint: x + v_pred

    return x.reshape(n_samples, B, 43, 6)  # hint: n_samples