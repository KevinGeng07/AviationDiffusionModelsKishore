import torch
import torch.nn as nn
import math

# ─────────────────────────────────────────────────────────────────────────────
# 1. Sinusoidal embedding — encodes a scalar (like timestep t) into a vector
# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_embedding(t, d_model):
    """
    t:       (B,) float tensor — diffusion timestep
    returns: (B, d_model) tensor
    """
    device = t.device
    half   = d_model // 2

    # Exponentially spaced frequencies
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / half
    )

    # Outer product: each timestep × each frequency
    args = t[:, None] * freqs[None, :]          # (B, half)

    # Concatenate sin and cos
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
    return embedding


def sinusoidal_embedding_seq(x, d_model):
    """
    x:       (B, T) float tensor — e.g. t_rel timestamps for all 86 positions
    returns: (B, T, d_model) tensor
    """
    device = x.device
    half   = d_model // 2

    freqs  = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / half
    )

    args   = x.unsqueeze(-1) * freqs[None, None, :]    # (B, T, half)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, T, d_model)
    return embedding


# ─────────────────────────────────────────────────────────────────────────────
# 2. AdaLN block — transformer block conditioned on diffusion time via
#    adaptive layer norm (scale + shift from conditioning vector)
# ─────────────────────────────────────────────────────────────────────────────
class AdaLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        # Layer norms (no affine — AdaLN supplies scale/shift instead)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Self attention
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN projection: conditioning vector → 4 vectors (s1, b1, s2, b2)
        # initialized to zero so conditioning is identity at start of training
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond, mask=None):
        """
        x:    (B, T, d_model)
        cond: (B, d_model)     — conditioning vector from diffusion timestep
        mask: (T, T) bool      — attention mask (True = ignore)
        """
        # Split conditioning into 4 scale/shift vectors
        s1, b1, s2, b2 = self.adaLN(cond).chunk(4, dim=-1)  # each (B, d_model)

        # Unsqueeze for broadcasting across sequence dimension
        s1, b1 = s1.unsqueeze(1), b1.unsqueeze(1)           # (B, 1, d_model)
        s2, b2 = s2.unsqueeze(1), b2.unsqueeze(1)

        # Attention with AdaLN
        h = self.norm1(x) * (1 + s1) + b1
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out                       # hint: residual connection

        # Feedforward with AdaLN
        h = self.norm2(x) * (1 + s2) + b2
        x = x + self.ff(h)                     # hint: residual connection

        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full DiT model
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryDiT(nn.Module):
    def __init__(self,
                 obs_len  = 43,
                 fut_len  = 43,
                 in_dim   = 6,
                 d_model  = 256,
                 n_heads  = 8,
                 n_layers = 6,
                 dropout  = 0.1):
        super().__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len
        seq_len      = obs_len + fut_len   # 86 total tokens

        # ── Input projection: 6 features → d_model for every token ──────────
        self.input_proj = nn.Linear(in_dim, d_model)  # hint: in_dim → d_model

        # ── Learnable positional embedding: one vector per position (0..85) ─
        self.pos_emb = nn.Embedding(seq_len, d_model)          # hint: seq_len positions

        # ── t_rel MLP: projects sinusoidal t_rel embedding → d_model ────────
        self.trel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # ── Flow time MLP: projects sinusoidal t embedding → d_model ────────
        # This becomes the AdaLN conditioning vector
        self.tflow_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),                        # hint: d_model → d_model
        )

        # ── Stack of AdaLN transformer blocks ───────────────────────────────
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)                             # hint: n_layers blocks
        ])

        # ── Output head: project future tokens → 6 features (predicted noise)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, in_dim)          # hint: → in_dim

        # ── Attention mask: obs tokens cannot attend to future tokens ────────
        # Shape (seq_len, seq_len) — True means IGNORE that position
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask[:obs_len, obs_len:] = True                      # hint: True
        self.register_buffer("attn_mask", mask)

    def forward(self, x_obs, x_t, t, t_rel):
        """
        x_obs:  (B, 43, 6)  — normalized observed context
        x_t:    (B, 43, 6)  — noisy future trajectory
        t:      (B,)        — diffusion timestep (integer, 0..T-1)
        t_rel:  (B, 86)     — normalized relative timestamps for all 86 steps

        returns: (B, 43, 6) — predicted noise on future tokens only
        """
        B   = x_obs.shape[0]
        device = x_obs.device

        # Step 1: project obs and noisy future to d_model, concatenate
        tokens = torch.cat([
            self.input_proj(x_obs),    # (B, 43, d_model)
            self.input_proj(x_t),      # (B, 43, d_model)
        ], dim=1)         # hint: concatenate along sequence dimension (dim=1)
        # tokens is now (B, 86, d_model)

        # Step 2: add learnable positional embedding
        positions = torch.arange(self.obs_len + self.fut_len, device=device)  # (86,)
        tokens = tokens + self.pos_emb(positions)                                        # hint: self.pos_emb(positions)

        # Step 3: add t_rel sinusoidal embedding through MLP
        trel_emb = sinusoidal_embedding_seq(t_rel, tokens.shape[-1])           # (B, 86, d_model)
        tokens   = tokens + self.trel_mlp(trel_emb)                      # hint: trel_emb

        # Step 4: build AdaLN conditioning vector from diffusion timestep
        # multiply t by 1000 to spread values (same trick as mentor's code)
        cond = self.tflow_mlp(
            sinusoidal_embedding(t.float() * 1000.0, tokens.shape[-1]) # hint: * 1000.0
        )   # (B, d_model)

        # Step 5: pass through all transformer blocks
        for block in self.blocks:
            tokens = block(tokens, cond, mask=self.attn_mask)                  # hint: self.attn_mask

        # Step 6: output head on future tokens only
        fut_tokens = tokens[:, self.obs_len:, :]                               # (B, 43, d_model)
        fut_tokens = self.output_norm(fut_tokens)                          # hint: fut_tokens
        noise_pred = self.output_head(fut_tokens)                          # hint: fut_tokens

        return noise_pred   # (B, 43, 6)
    
if __name__ == "__main__":
    model = TrajectoryDiT()

    B = 4
    x_obs = torch.randn(B, 43, 6)
    x_t   = torch.randn(B, 43, 6)
    t     = torch.randint(0, 1000, (B,))
    t_rel = torch.randn(B, 86)

    out = model(x_obs, x_t, t, t_rel)
    print(out.shape)   # should be torch.Size([4, 43, 6])

    from torchinfo import summary
    summary(model, input_data=(x_obs, x_t, t, t_rel))