import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal embedding — encodes diffusion/flow time t into a vector
# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_embedding(t, d_model):
    device = t.device
    half   = d_model // 2
    freqs  = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    args   = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE — Rotary Position Embedding (Option B: real timestamp positions)
# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, d_head):
        """
        d_head: dimension per attention head (d_model // n_heads)
        """
        super().__init__()
        half  = d_head // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, half).float() / half))
        self.register_buffer("freqs", freqs)

    def get_rotations(self, positions):
        """
        CHANGED vs Option A: positions is now (B, T) not (T,)
        because each sample has different real timestamps.

        positions: (B, T) — t_rel values e.g. 0.0, 3.2, 6.5, ..., 254.1
        returns:   cos (B, T, half), sin (B, T, half)
        """
        # (B, T) → (B, T, half) via unsqueeze + broadcast
        angles = positions.unsqueeze(-1) * self.freqs   # (B, T, half)
        return torch.cos(angles), torch.sin(angles)

    def rotate(self, x, cos, sin):
        """
        CHANGED vs Option A: cos/sin are now (B, T, half) not (T, half)
        so we unsqueeze differently for broadcasting.

        x:   (B, T, n_heads, d_head)
        cos: (B, T, half)
        sin: (B, T, half)
        returns: (B, T, n_heads, d_head) rotated
        """
        x1 = x[..., :x.shape[-1] // 2]   # (B, T, n_heads, half)
        x2 = x[..., x.shape[-1] // 2:]   # (B, T, n_heads, half)

        # CHANGED: (B, T, half) → (B, T, 1, half) — only need dim 2 unsqueeze
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE-aware self-attention (Option B: batch-level positions)
# ─────────────────────────────────────────────────────────────────────────────
class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.rope     = RotaryEmbedding(self.d_head)

    def forward(self, x, positions, mask=None):
        """
        CHANGED vs Option A: positions is now (B, T) real timestamps
        instead of (T,) token indices.

        x:         (B, T, d_model)
        positions: (B, T) — normalized t_rel timestamps
        mask:      (T, T) bool — True means ignore
        """
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head)

        # CHANGED: positions is (B, T) — each sample has its own timestamps
        cos, sin = self.rope.get_rotations(positions)   # (B, T, half) each
        Q = self.rope.rotate(Q, cos, sin)
        K = self.rope.rotate(K, cos, sin)

        # Reshape for attention
        Q = Q.permute(0, 2, 1, 3)   # (B, n_heads, T, d_head)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        scale  = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, V)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# AdaLN block — identical to Option A, just passes (B, T) positions through
# ─────────────────────────────────────────────────────────────────────────────
class AdaLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = RoPEAttention(d_model, n_heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond, positions, mask=None):
        """
        x:         (B, T, d_model)
        cond:      (B, d_model)
        positions: (B, T) — real t_rel timestamps passed to RoPEAttention
        mask:      (T, T) bool
        """
        s1, b1, s2, b2 = self.adaLN(cond).chunk(4, dim=-1)
        s1, b1 = s1.unsqueeze(1), b1.unsqueeze(1)
        s2, b2 = s2.unsqueeze(1), b2.unsqueeze(1)

        h        = self.norm1(x) * (1 + s1) + b1
        attn_out = self.attn(h, positions, mask=mask)
        x        = x + attn_out

        h = self.norm2(x) * (1 + s2) + b2
        x = x + self.ff(h)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full DiT with RoPE Option B (real timestamp positions)
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
        seq_len      = obs_len + fut_len

        self.input_proj = nn.Linear(in_dim, d_model)

        # REMOVED: pos_emb, trel_mlp
        # Option B uses t_rel directly as RoPE rotation angles
        # so no separate MLP needed to process timing
        self.tflow_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, in_dim)

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask[:obs_len, obs_len:] = True
        self.register_buffer("attn_mask", mask)

    def forward(self, x_obs, x_t, t, t_rel):
        """
        x_obs:  (B, 43, 6)  normalized observed context
        x_t:    (B, 43, 6)  noisy/interpolated future
        t:      (B,)        diffusion/flow time
        t_rel:  (B, 86)     normalized relative timestamps
                            CHANGED vs Option A: actually used here as RoPE positions

        returns: (B, 43, 6) predicted noise or velocity
        """
        device = x_obs.device

        # Step 1: project and concatenate tokens
        tokens = torch.cat([
            self.input_proj(x_obs),
            self.input_proj(x_t),
        ], dim=1)   # (B, 86, d_model)

        # Step 2: CHANGED vs Option A
        # positions = t_rel (B, 86) — real timestamps per sample
        # each token's rotation angle reflects its actual time offset
        # so attention between tokens encodes real time gaps
        positions = t_rel   # (B, 86)

        # Step 3: AdaLN conditioning from diffusion/flow time
        cond = self.tflow_mlp(
            sinusoidal_embedding(t.float() * 1000.0, tokens.shape[-1])
        )   # (B, d_model)

        # Step 4: pass through all transformer blocks
        for block in self.blocks:
            tokens = block(tokens, cond, positions, mask=self.attn_mask)

        # Step 5: output head on future tokens only
        fut_tokens = tokens[:, self.obs_len:, :]
        fut_tokens = self.output_norm(fut_tokens)
        return self.output_head(fut_tokens)   # (B, 43, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Quick shape check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = TrajectoryDiT()
    B     = 4
    x_obs = torch.randn(B, 43, 6)
    x_t   = torch.randn(B, 43, 6)
    t     = torch.rand(B)
    t_rel = torch.randn(B, 86)

    out = model(x_obs, x_t, t, t_rel)
    print(out.shape)   # should be torch.Size([4, 43, 6])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")