# AviationDiffusionModelsKishore

Probabilistic **aircraft trajectory prediction** for the San Francisco Bay Area using
generative models. Given ~2 minutes of observed ADS-B track for an aircraft, the model
samples many plausible ~2-minute futures. Two generative frameworks are implemented on
top of the same transformer backbone:

- **DDIM** — denoising diffusion (predict the added noise, then iteratively denoise)
- **CFM** — conditional flow matching / rectified flow (predict a velocity field, then integrate an ODE)

and the backbone (a **Diffusion Transformer, DiT**) comes in four variants: the baseline
with learned positions, two **RoPE** variants, and **KEVIN_DiT** (SwiGLU feed-forward).

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Setup](#2-setup)
3. [Data pipeline](#3-data-pipeline)
4. [The prediction task](#4-the-prediction-task)
5. [Model architecture: the Trajectory DiT](#5-model-architecture-the-trajectory-dit)
6. [RoPE — rotary position embeddings](#6-rope--rotary-position-embeddings)
7. [DDIM — denoising diffusion](#7-ddim--denoising-diffusion)
8. [CFM — conditional flow matching](#8-cfm--conditional-flow-matching)
9. [Training](#9-training)
10. [Evaluation and metrics](#10-evaluation-and-metrics)
11. [Prediction Explorer UI](#11-prediction-explorer-ui)
12. [Static prediction maps](#12-static-prediction-maps)
13. [Gotchas and known issues](#13-gotchas-and-known-issues)

---

## 1. Repository layout

```
.
├── Data/
│   ├── ADSBnetcdfbuilder.py        # raw ADS-B CSVs  →  windowed .nc dataset
│   ├── adsb/                       # raw CSVs: flight_log-0.csv … flight_log-36.csv (gitignored)
│   └── trajectories_adsblol_seq86_stage2.nc   # the training dataset (gitignored)
│
├── DataLoaders/
│   ├── ADSBdataset.py              # load .nc, normalize, split by aircraft, PyTorch DataLoaders
│   └── sanityCheckMathurinNetCDF.py# print the variables/attributes of a .nc (+ CUDA check)
│
├── models/
│   ├── dit.py                      # baseline Trajectory DiT (learned pos-emb + t_rel embedding)
│   ├── dit_RoPE_original.py        # DiT with RoPE over token indices          ("RoPE-A")
│   ├── dit_RoPE_timestamps.py      # DiT with RoPE over real timestamps        ("RoPE-B")
│   ├── KEVIN_DiT.py                # baseline DiT with SwiGLU feed-forward
│   ├── ddim.py                     # cosine noise schedule + forward diffusion
│   └── cfm.py                      # flow-time sampling, linear interpolation, Euler sampler
│
├── training/
│   ├── trainDDIMBaseLocal.py       # DDIM + baseline DiT, local run (hardcoded paths)
│   ├── trainDDIMBaseSAVIO.py       # DDIM + baseline DiT, CLI args (for Savio)
│   ├── trainCFMbaseLocal.py        # CFM  + baseline DiT, local run
│   ├── trainCFMbaseSavio.py        # CFM  + baseline DiT, CLI args (for Savio)
│   ├── trainCFMRoPEOG.py           # CFM  + RoPE-A DiT, local run
│   ├── trainCFMRoPEtimestamps.py   # CFM  + RoPE-B DiT, local run
│   ├── trainCFMKevinDIT.py         # CFM  + KEVIN_DiT, local run
│   └── trainCFMKevinDITSavio.py    # CFM  + KEVIN_DiT, CLI args (for Savio)
│
├── savio_batchfiles/
│   ├── DDIMBaseSavio7MParameterModel.sh   # SLURM job for trainDDIMBaseSAVIO.py
│   ├── CFMBaseSavio7MParameterModel.sh    # SLURM job for trainCFMbaseSavio.py
│   └── CFMKevinDITSavio7MParameterModel.sh# SLURM job for trainCFMKevinDITSavio.py
│
├── eval/
│   └── visualization.py            # render DDIM predictions to folium HTML maps
│
├── ui/
│   ├── app.py                      # Flask "Prediction Explorer" backend
│   ├── templates/index.html        # Leaflet map front-end
│   └── requirements.txt
│
├── checkpoints_cfm_kevin_dit/      # best_kevin_dit.pt, last_kevin_dit.pt (gitignored)
├── predictions_ep49_100k/          # pre-rendered DDIM maps (epoch 49, 100k-sample run)
├── predictions_ep100_100k/         #   … epoch 100, 100k-sample run
├── predictions_ep100_full/         #   … epoch 100, full-dataset run
└── predictions.html                # single pre-rendered DDIM map
```

---

## 2. Setup

Python 3.10+ (developed on 3.13). From the repo root:

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy pandas netCDF4 tqdm flask
pip install folium      # only for eval/visualization.py
pip install torchinfo   # only for the `python models/dit.py` shape check
```

A GPU is used automatically when `torch.cuda.is_available()`; everything (including the UI)
also runs on CPU, just slower. Training on the full dataset is meant for a GPU (Savio).

---

## 3. Data pipeline

### 3.1 Raw data

`Data/adsb/flight_log-N.csv` are snapshots of every aircraft visible around the Bay Area,
one row per aircraft per poll (≈3 s cadence). The files are **chronological in numeric
order** (`0, 1, 2, … 36`), covering roughly 2026-04-29 → 2026-05-16. Columns used:

| column | meaning |
|---|---|
| `time_position` | Unix time of the position fix |
| `icao24` | 24-bit aircraft transponder address (unique per airframe) |
| `callsign` | flight number / registration |
| `longitude`, `latitude`, `geo_altitude` | position (deg, deg, m) |
| `velocity`, `true_track`, `vertical_rate` | ground speed (m/s), heading (deg), climb rate (m/s) |
| `on_ground` | bool |

### 3.2 Building the `.nc` dataset — `Data/ADSBnetcdfbuilder.py`

```bash
venv/bin/python Data/ADSBnetcdfbuilder.py \
    $(ls Data/adsb/flight_log-*.csv | sort -t- -k3 -n) \
    Data/trajectories_adsblol_seq86_stage2.nc \
    --seq-len 86 --offset 10
```

> **Pass files in numeric order.** A plain `flight_log-*.csv` glob sorts lexically
> (`0, 1, 10, 11, …`), which breaks the cross-file stitching described below.
> With `--offset 10` the full build takes ~30 min and peaks at a few GB of RAM;
> `--offset 1` produces ~10× more samples and will not fit in 16 GB.

Steps, in order:

1. **`load_file`** — keep the 10 columns above, drop rows with no time or velocity,
   fill missing altitude / vertical rate with 0, forward/back-fill callsigns per aircraft,
   de-duplicate `(icao24, time_position)`.
2. **`convertToCartesian`** — project to a local metric frame centred on **SFO
   (37.6213°N, 122.3790°W)** using an equirectangular approximation:
   - `x = (R + alt)·cos(lat)·(lon − lon₀)` (east, m)
   - `y = R·(lat − lat₀)` (north, m)
   - `z = alt` (m)
   - `vx = v·sin(track)`, `vy = v·cos(track)`, `vz = vertical_rate` (m/s)
3. **Stitching** — files are processed one at a time; the last `seq_len − 1` timestamps
   of each file are prepended to the next so flights crossing a file boundary still
   produce windows.
4. **`segment_aircraft`** — split each aircraft's track into continuous segments; a new
   segment starts on a gap > 120 s or a callsign change. Segments shorter than `seq_len`
   or entirely on the ground are dropped.
5. **`extract_windows`** — slide an 86-step window along each segment with stride
   `offset`. Windows that are all on-ground or dip below −50 m altitude are discarded.
6. **`compute_stats`** — global statistics used later for normalization (see below).
7. **`write_netcdf`** — write everything to NetCDF4.

**What `offset` means.** It is the stride of the sliding window: with `offset=10`,
consecutive windows of the same aircraft start 10 pings (~30 s) apart and share 76 of
their 86 points. Smaller offsets give more (but highly redundant) samples; dataset size
scales as `1/offset`. The UI also relies on this stride to chain windows into longer
ground truth (§11).

### 3.3 `.nc` schema

| variable | shape | description |
|---|---|---|
| `trajectory` | `(N, 86, 6)` float32 | **absolute** `x, y, z, vx, vy, vz` in metres / m·s⁻¹ from SFO |
| `timestamps` | `(N, 86)` float64 | Unix time of each step |
| `icao24`, `callsign` | `(N,)` str | aircraft identity of each window |
| `origin` | `(N, 2)` float32 | `x, y` at step 0 (builder output only) |

Global attributes: `feature_mean`, `feature_std` (6-vectors, over all absolute values),
`t_rel_mean`, `t_rel_std` (time since window start), `dt_mean`, `dt_std`, `delta_mean`,
`delta_std`, `seq_len`, `offset`, `description`.

The current dataset has **1,349,388 windows** (`offset = 10`).

Inspect any `.nc` with:

```bash
venv/bin/python -c "import netCDF4 as nc; d=nc.Dataset('Data/trajectories_adsblol_seq86_stage2.nc'); print(d); print(d['trajectory'])"
```

### 3.4 Loading for training — `DataLoaders/ADSBdataset.py`

- **`load_netcdf`** reads the whole `trajectory` / `timestamps` arrays into RAM (~4 GB for the full file) plus the stats.
- **`split_by_icao`** splits by **aircraft, not by window**: 85 % / 10 % / 5 % of unique
  `icao24`s go to train / val / test (seeded with `seed=42` in `get_dataloaders`). This
  prevents near-duplicate overlapping windows of the same flight leaking across splits.
- **`normalize`** applies a per-feature z-score: `(traj − feature_mean) / feature_std`.
- **`TrajectoryDataset.__getitem__`** returns
  - `obs`   — normalized steps 0‥42 `(43, 6)`
  - `fut`   — normalized steps 43‥85 `(43, 6)`
  - `t_rel` — `(timestamps − timestamps[0])`, z-scored with `t_rel_mean/std` `(86,)`
- **`get_dataloaders(path, batch_size, subset=None)`** — optionally truncate to the first
  `subset` windows (the local scripts use `subset=100_000`).

---

## 4. The prediction task

```
     observed (context)                    future (generated)
 ┌──────────────────────────────┐┌──────────────────────────────┐
 step 0 ………………………………………… step 42 step 43 ………………………………………… step 85
 ≈ 2 min of real track            ≈ 2 min the model samples
```

- **Input:** 43 normalized states `(x, y, z, vx, vy, vz)` + the relative timestamps of all 86 steps.
- **Output:** 43 future states. Because the model is generative, every call with fresh
  noise yields a *different plausible future*; drawing K samples gives a distribution
  (e.g. holding vs. turning onto final approach).
- ADS-B pings are irregular (mean gap ≈ 3.1 s, std ≈ 2.6 s), which is why `t_rel` is fed
  to the model and why the RoPE-B variant exists.

---

## 5. Model architecture: the Trajectory DiT

All variants expose the same interface, so any of them can be trained with either DDIM or CFM:

```python
model(x_obs, x_t, t, t_rel) -> (B, 43, 6)
# x_obs: (B, 43, 6) clean observed context      (normalized)
# x_t:   (B, 43, 6) noisy / interpolated future (the thing being generated)
# t:     (B,)       diffusion / flow time in [0, 1)
# t_rel: (B, 86)    normalized relative timestamps
# out:   DDIM → predicted noise ε ;  CFM → predicted velocity v
```

A **Diffusion Transformer** (Peebles & Xie, 2023) is a plain transformer whose layers are
conditioned on the diffusion time through *adaptive layer norm*. Here the "image patches"
are trajectory time steps.

### 5.1 Forward pass (`models/dit.py`)

```
 x_obs (B,43,6) ─┐                                       t (B,)
                 ├─ input_proj: Linear(6→256) ─┐          │
 x_t   (B,43,6) ─┘                             │   sinusoidal_embedding(t·1000)
                                               ▼          │
                              tokens (B, 86, 256)    tflow_mlp: Linear–SiLU–Linear
                                + pos_emb[0..85]          │
                                + trel_mlp(sinusoid(t_rel))│
                                               │          ▼
                                               │     cond (B, 256)
                                 ┌─────────────▼──────────┴──────────┐
                                 │  AdaLNBlock × n_layers (6)        │
                                 │   (masked self-attention + FFN)   │
                                 └─────────────┬─────────────────────┘
                                               │ keep tokens 43..85 only
                                     LayerNorm → Linear(256→6)
                                               ▼
                                   ε̂ or v̂   (B, 43, 6)
```

1. **Tokenization.** Every time step — observed *and* noisy future — becomes one token via
   a shared `Linear(6 → d_model)`. Concatenating gives 86 tokens.
2. **Position.** A learned `nn.Embedding(86, d_model)` marks *which slot* each token is in
   (and so whether it is context or future).
3. **Real timing.** `t_rel` (seconds since window start, normalized) is turned into a
   per-token sinusoidal embedding (`sinusoidal_embedding_seq`) and passed through
   `trel_mlp`, so the model knows the actual time gaps between irregular pings.
4. **Diffusion-time conditioning.** The scalar `t` (multiplied by 1000 to spread
   frequencies) goes through a sinusoidal embedding and `tflow_mlp`, producing one
   conditioning vector `cond` per sample.
5. **Transformer blocks** (below), then
6. **Output head** — only the 43 future tokens are normalized and projected back to 6 features.

### 5.2 The AdaLN block

```python
s1, b1, s2, b2 = adaLN(cond).chunk(4)          # adaLN = SiLU → Linear(d → 4d)

h = LayerNorm(x) * (1 + s1) + b1               # no learned affine in the LayerNorm
x = x + MultiheadAttention(h, h, h, mask)      # residual
h = LayerNorm(x) * (1 + s2) + b2
x = x + FFN(h)                                 # residual; FFN = Linear(d→4d)–GELU–Linear(4d→d)
```

- **Adaptive LayerNorm**: instead of fixed LayerNorm scale/shift, they are *predicted
  from the diffusion time*. Early in denoising (very noisy input) the network can behave
  very differently than late in denoising, with a single set of weights.
- The final `adaLN` linear is **zero-initialised**, so every block starts as an identity-
  conditioned standard transformer block (the "adaLN-Zero"-style trick for stable training).

### 5.3 The attention mask

```python
mask[:43, 43:] = True     # True = may NOT attend
```

| query ↓ / key → | obs tokens | future tokens |
|---|---|---|
| **obs tokens**    | ✅ | ❌ |
| **future tokens** | ✅ | ✅ |

Observed tokens only attend to each other, so the encoding of the context is independent
of the current noisy guess; future tokens attend to everything (context + each other),
which lets them both read the history and stay mutually consistent.

### 5.4 Default hyper-parameters

`d_model = 256`, `n_heads = 8` (32 dims/head), `n_layers = 6`, `dropout = 0.1`,
`obs_len = fut_len = 43`, `in_dim = 6`.

### 5.5 Variants

| file | position / timing signal | feed-forward | params (6 layers) |
|---|---|---|---|
| `dit.py` | learned `pos_emb` + `t_rel` sinusoid MLP (added to tokens) | GELU MLP, 4× | 6.60 M |
| `dit_RoPE_original.py` (RoPE-A) | RoPE with angles = token index 0‥85 | GELU MLP, 4× | 6.45 M |
| `dit_RoPE_timestamps.py` (RoPE-B) | RoPE with angles = normalized `t_rel` | GELU MLP, 4× | 6.45 M |
| `KEVIN_DiT.py` | same as `dit.py` | **SwiGLU, 3×, no bias** | 7.25 M |

The RoPE models drop `pos_emb` and `trel_mlp` (position lives inside attention instead —
§6) and replace `nn.MultiheadAttention` with a hand-written `RoPEAttention`.

**KEVIN_DiT** swaps the feed-forward for **SwiGLU**:

```python
FFN(x) = W_down( SiLU(W_gate x) ⊙ (W_up x) )     # W_gate, W_up: d→3d ;  W_down: 3d→d
```

A gated linear unit lets the network learn multiplicative feature interactions; using
3× instead of 4× width keeps the parameter count similar to the GELU MLP (3 matrices
instead of 2). It also defines an extra top-level `self.adaLN` that is never used in
`forward` (≈0.26 M dead parameters — kept because the checkpoint contains it). The
included checkpoint `checkpoints_cfm_kevin_dit/` is a **6-layer, CFM-trained** KEVIN_DiT
(epoch 84, val minFDE 824.5 m), trained with `training/trainCFMKevinDIT*.py`.

Shape-check any model with `venv/bin/python models/dit.py` (needs `torchinfo`) or
`venv/bin/python models/dit_RoPE_original.py`.

---

## 6. RoPE — rotary position embeddings

**Idea.** Instead of *adding* a position vector to each token, RoPE (Su et al., 2021)
*rotates* the query and key vectors inside attention by an angle proportional to the
token's position. Split each head's 32-dim vector into 16 pairs; pair *i* is rotated by
angle `pos · θᵢ`, with frequencies `θᵢ = 10000^(−i/16)`:

```
[x₁']   [cos(pos·θᵢ)  −sin(pos·θᵢ)] [x₁]
[x₂'] = [sin(pos·θᵢ)   cos(pos·θᵢ)] [x₂]
```

(`RotaryEmbedding.rotate` implements this with the "first half / second half" pairing.)

**Why it helps.** The dot product of a rotated query at position *m* and a rotated key at
position *n* depends only on **m − n**:

```
⟨R(m)q, R(n)k⟩ = ⟨q, R(n − m)k⟩
```

So attention scores natively encode *relative* distance between tokens, at every layer,
with no learned position parameters. Low-frequency pairs capture long-range offsets,
high-frequency pairs capture fine offsets. Values (V) are not rotated.

**The two variants here**

| | RoPE-A (`dit_RoPE_original.py`) | RoPE-B (`dit_RoPE_timestamps.py`) |
|---|---|---|
| `positions` | `arange(86)` — shared `(86,)` | `t_rel` — per-sample `(B, 86)` |
| relative quantity encoded | "how many pings apart" | "how many (normalized) seconds apart" |
| uses `t_rel`? | no (kept in signature only) | yes — it *is* the position |

RoPE-B is the physically motivated one: ADS-B pings are irregular, so two tokens 10 pings
apart might be 20 s or 60 s apart; with RoPE-B the attention geometry reflects real time.
Note that `t_rel` is z-scored, so the rotation angles are on a small scale; the frequency
spectrum therefore effectively uses the low-θ (long-range) end differently than RoPE-A.

---

## 7. DDIM — denoising diffusion

Files: `models/ddim.py`, `training/trainDDIM*.py`, sampler in `ui/app.py` /
`eval/visualization.py`.

### 7.1 Forward (noising) process

A **cosine schedule** (Nichol & Dhariwal, 2021) with `T = 1000` defines how much signal
remains at step *t*:

```
ᾱ_t = cos²( (t/T + s)/(1 + s) · π/2 ),  s = 0.008        (then β clamped to [1e-4, 0.999])
```

Noise can be added in closed form at any step:

```
x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε,      ε ~ N(0, I)
```

where `x₀` is the clean normalized future `(B, 43, 6)`. `t = 0` is ~clean data,
`t = T−1` is ~pure noise.

### 7.2 Training (ε-prediction)

```python
t          = randint(0, T)                         # uniform
x_t, noise = forward_diffusion(fut, t, alphas_cumprod)
noise_pred = model(obs, x_t, t / T, t_rel)         # model sees t in [0, 1)
loss       = MSE(noise_pred, noise)
```

The network learns to identify *which noise was added*, given the context and the noise level.

### 7.3 Sampling (DDIM, η = 0)

DDIM (Song et al., 2021) is a deterministic sampler that can skip steps. With 20 steps,
the timesteps visited are `950, 900, …, 50, 0`. At each one:

```python
ε̂      = model(obs, x, t/T, t_rel)
x̂₀     = (x − √(1−ᾱ_t) · ε̂) / √ᾱ_t           # predicted clean trajectory
x      = √ᾱ_prev · x̂₀ + √(1−ᾱ_prev) · ε̂       # jump to the previous (less noisy) step
```

starting from `x ~ N(0, I)` and ending with `ᾱ_prev = 1` (fully clean). Different
starting noise ⇒ different futures. In the UI, the "steps" slider controls this count.

---

## 8. CFM — conditional flow matching

Files: `models/cfm.py`, `training/trainCFM*.py`.

### 8.1 Probability path

CFM / rectified flow (Lipman et al., 2023; Liu et al., 2023) connects noise and data with
a **straight line**. Note the time direction is the *opposite* of DDIM: here
`t = 0` is noise and `t = 1` is data.

```
x_t   = (1 − t) · ε + t · x₀          ε ~ N(0, I)
v_tgt = dx_t/dt = x₀ − ε              (constant along the line)
```

### 8.2 Training (velocity prediction)

```python
t         = sigmoid(N(1, 1))                         # logit-normal flow time
x_t, ε, v = forward_cfm(fut, t)
v_pred    = model(obs, x_t, t, t_rel)

w         = ones(43); w[-10:] = 2.0                  # last quarter of the future ×2
loss      = mean( w[None, :, None] · (v_pred − v)² )
```

- **Logit-normal time** (the SD3 trick): `t = sigmoid(u)`, `u ~ N(1, 1)`. Training effort
  concentrates in the middle-to-late part of the path (median t ≈ 0.73) where the
  velocity is hardest to predict, instead of uniformly over `[0, 1]`.
- **Endpoint weighting**: the last 10 of 43 future steps get double weight, since the
  final position (FDE) is the headline metric and errors accumulate along the horizon.

### 8.3 Sampling (Euler ODE)

```python
x = randn(...)                          # t = 0
for k in range(n_steps):                # n_steps = 20
    v = model(obs, x, k / n_steps, t_rel)
    x = x + v / n_steps                 # Euler step toward t = 1
```

Because the learned paths are close to straight, a simple first-order integrator with
~20 steps is enough.

### 8.4 DDIM vs CFM at a glance

| | DDIM | CFM |
|---|---|---|
| path noise → data | curved (cosine schedule, variance-preserving) | straight line |
| network predicts | noise ε | velocity v = x₀ − ε |
| time convention | t = T noise → 0 data | t = 0 noise → 1 data |
| training t | uniform integer 0…999 | logit-normal in (0, 1) |
| loss | plain MSE | MSE, last ¼ of horizon ×2 |
| sampler | deterministic DDIM update, 20 steps | Euler ODE, 20 steps |
| validation metric | FDE of 1 sample | minFDE over 5 samples |

---

## 9. Training

### 9.1 Common setup (all six scripts)

- **Optimizer:** AdamW, `lr = 1e-4`, `weight_decay = 0.01`, grad-norm clip 1.0.
- **Schedule:** linear warm-up for 1000 steps, then cosine annealing over the epochs.
- **EMA:** an exponential moving average of the weights (`decay = 0.9999`) is kept and is
  what validation, the UI and the visualizer use (`ema_state`).
- **Batch size** 64; **epochs** 100.
- **Checkpoints** (`.pt` dicts): `epoch`, `model_state`, `ema_state`, `optimizer_state`,
  `val_fde`. `last.pt` is written every epoch (training auto-resumes from it), `best.pt`
  whenever validation FDE improves.
- **Validation:** CFM scripts print a *rough* minFDE (5 val batches) every epoch and a
  *proper* one (20 batches ≈ 1280 windows) every 5 epochs; DDIM scripts validate every 5 epochs.

### 9.2 Local runs

The `*Local.py`, `trainCFMRoPEOG.py` and `trainCFMRoPEtimestamps.py` scripts have their
settings hardcoded in the `if __name__ == "__main__":` block — notably a Windows path
`nc_path = r"D:\trajectories_adsblol_seq86_stage2.nc"` and `subset = 100_000`. Edit those
lines, then:

```bash
venv/bin/python training/trainCFMbaseLocal.py         # → checkpoints_cfm/
venv/bin/python training/trainDDIMBaseLocal.py        # → checkpoints/
venv/bin/python training/trainCFMRoPEOG.py            # → checkpoints_rope_a/
venv/bin/python training/trainCFMRoPEtimestamps.py    # → checkpoints_rope_b/
venv/bin/python training/trainCFMKevinDIT.py          # → checkpoints_cfm_kevin_dit/
```

### 9.3 CLI runs / Savio

```bash
venv/bin/python training/trainCFMbaseSavio.py \
    --nc_path Data/trajectories_adsblol_seq86_stage2.nc \
    --output_dir checkpoints_cfm --epochs 100 --batch_size 64 \
    [--d_model 256] [--n_layers 6] [--subset 100000]

venv/bin/python training/trainDDIMBaseSAVIO.py   --nc_path ... --output_dir checkpoints
venv/bin/python training/trainCFMKevinDITSavio.py --nc_path ... --output_dir checkpoints_cfm_kevin_dit
```

On the Berkeley **Savio** cluster, submit the SLURM wrappers (1× GTX 2080 Ti, 20 h,
conda env `adsb`, paths under `/global/scratch/users/kishore26/adsb-diffusion/`):

```bash
sbatch savio_batchfiles/CFMBaseSavio7MParameterModel.sh
sbatch savio_batchfiles/DDIMBaseSavio7MParameterModel.sh
sbatch savio_batchfiles/CFMKevinDITSavio7MParameterModel.sh
# logs → logs/cfm_<jobid>.out / logs/ddim_<jobid>.out
```

Edit `NC_PATH`, `OUTPUT_DIR` and the `cd` line for your own scratch directory.

### 9.4 Training a different backbone

Every script picks its model with a single import, e.g. in `trainCFMbaseSavio.py`:

```python
from models.dit import TrajectoryDiT          # swap for models.KEVIN_DiT / dit_RoPE_* …
```

Constructor arguments are passed explicitly (`n_layers=6` by default), so a class's own
default `n_layers` is not what gets trained unless you change the script/CLI flag.

---

## 10. Evaluation and metrics

- **FDE (final displacement error)** — Euclidean distance in metres between predicted and
  true `(x, y)` at the last future step. Normalized predictions are rescaled with
  `feature_std[:2]` (the mean cancels in the difference).
- **minFDE@K** — draw K futures, keep the best one's FDE, average over windows. This is
  the standard metric for multimodal predictors (it rewards covering the true outcome).
  CFM validation uses K = 5; DDIM validation uses a single sample.
- The best validation score is stored in each checkpoint:

```bash
venv/bin/python -c "import torch; c=torch.load('checkpoints_cfm_kevin_dit/best_kevin_dit.pt', map_location='cpu', weights_only=False); print(c['epoch'], c['val_fde'])"
```

- **ADE (average displacement error)** — mean `(x, y)` distance over every future step,
  not just the last one. Reported per sample in the UI.
- **KDE-NLL** — fit a Gaussian kernel density (Scott's-rule bandwidth) to the K sampled
  positions at each step, evaluate the negative log-likelihood of the true position, and
  average over the horizon (nats; lower = the predicted distribution is better
  calibrated). Unlike min/mean FDE this rewards spreading samples where the truth
  actually is, and penalizes both over-confident and over-dispersed predictions.
- Per-flight min / mean / max FDE and ADE plus KDE-NLL are shown live in the UI (§11).

There is currently no script that evaluates on the held-out **test** split.

---

## 11. Prediction Explorer UI

An interactive Leaflet map for picking a flight, running any trained model on it, and
comparing the sampled futures with the ground truth.

```bash
venv/bin/python ui/app.py --nc Data/trajectories_adsblol_seq86_stage2.nc --port 5001
# open http://localhost:5001
```

Options: `--nc PATH` (or env `NC_PATH`), `--port` (default 5000), `--host`,
`--max-flights` (default 5000 — only the first N windows are loaded).

> **macOS:** port 5000 is taken by AirPlay Receiver, which answers `localhost:5000` with
> "Access to localhost was denied" (403). Use `--port 5001`, or disable AirPlay Receiver.

**Models** are listed in `MODEL_REGISTRY` in `ui/app.py`; an entry shows as *available*
only if its checkpoint file exists:

| id | checkpoint | type | backbone |
|---|---|---|---|
| `ddim` | `checkpoints/best.pt` | DDIM | `dit.py` |
| `cfm` | `checkpoints_cfm/last.pt` | CFM | `dit.py` |
| `cfm_rope_og` | `checkpoints_cfm_rope_og/best.pt` | CFM | RoPE-A |
| `cfm_rope_ts` | `checkpoints_cfm_rope_ts/best.pt` | CFM | RoPE-B |
| `cfm_kevin_dit` | `checkpoints_cfm_kevin_dit/best_kevin_dit.pt` | CFM | KEVIN_DiT (6 layers), shown as "CFMKevinDiT" |

To add a model: add a branch in `_build_model(arch)` that constructs it with the right
`n_layers`, and a registry entry with `ckpt`, `type` (`"ddim"`/`"cfm"`) and `arch`.

**Controls / behaviour**

- **Flights** are classified as *consumer* (airline-style callsign, e.g. `UAL123`) or *GA*
  (N-numbers, blank or other callsigns).
- **Samples** — number of futures drawn.
- **Steps** — DDIM sampling steps (CFM always uses 20 Euler steps).
- **Observation window** — use only the last *k* of the 43 observed points; earlier slots
  are padded by repeating the first visible point.
- **Total horizon** — predict beyond 43 steps by **autoregressive rollout**: each sampled
  43-step block becomes the context for the next block.
- **Extended ground truth** beyond 43 steps is built by chaining later windows of the
  same aircraft, each adding `offset` new points (requires consecutive windows in the file).
- **Metrics panel** — min / mean / max FDE and ADE across samples, plus KDE-NLL, up to the
  last step that has ground truth.
- **Prediction heatmap** (legend toggle) — density of the sampled trajectory endpoints.

API (JSON): `GET /api/status`, `GET /api/flights`, `GET /api/models`,
`POST /api/predict` with `{"flight_idx", "model", "n_samples", "n_steps", "total_horizon", "obs_window"}`.

---

## 12. Static prediction maps

`eval/visualization.py` loads a **DDIM** checkpoint with the baseline DiT, samples 20
futures for batches of validation windows and writes folium maps
(`predictions_N.html`: observed track, true future, sampled futures). Edit the
`nc_path`, `ckpt_path` and `output_dir` in its `__main__` block, then:

```bash
pip install folium
venv/bin/python eval/visualization.py
```

The committed `predictions_ep*/` folders and `predictions.html` are outputs of this
script; open them directly in a browser.

---

## 13. Gotchas and known issues

- **Hardcoded Windows paths** (`D:\trajectories_adsblol_seq86_stage2.nc`, `C:\dev\work\…`, `checkpoints\best.pt`)
  in the local training scripts, `eval/visualization.py` and
  `DataLoaders/sanityCheckMathurinNetCDF.py`. Change them to your local path.
- **`trajectory` must be absolute positions.** The dataloader normalizes `trajectory`
  with `feature_mean/std` and the UI plots it directly. `ADSBnetcdfbuilder.py` writes
  absolute `x, y, z, vx, vy, vz`; an earlier version wrote `dx, dy` deltas, which are
  incompatible with the trained models.
- **Normalization stats are not saved in checkpoints.** A model only behaves correctly
  with the `.nc` (and hence `feature_mean/std`, `t_rel_mean/std`) it was trained on.
- **Checkpoint folder names differ between training and UI:** the RoPE scripts write
  `checkpoints_rope_a/` and `checkpoints_rope_b/`, but the UI looks in
  `checkpoints_cfm_rope_og/` and `checkpoints_cfm_rope_ts/`. Rename or edit the registry.
- **Memory:** the dataloader loads the full `.nc` into RAM (~4 GB for 1.35 M windows);
  use `subset=` on small machines. The builder at `--offset 1` needs far more than 16 GB.
- **`num_workers=4, pin_memory=True`** in `get_dataloaders` can be slow or warn on
  macOS/CPU; lower `num_workers` if needed.
- **`sanityCheckMathurinNetCDF.py`** also calls `torch.cuda.get_device_name(0)`, which
  fails on machines without CUDA.
- **`.gitignore`** excludes `Data/adsb/`, `*.nc`, `*.csv`, `checkpoints/`, `checkpoints_cfm/`
  and `checkpoints_cfm_kevin_dit/`, but not the RoPE checkpoint folders — avoid committing
  them by accident (each `.pt` is ~110 MB, over GitHub's 100 MB file limit).

---

## References

- Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT), 2023
- Song, Meng & Ermon, *Denoising Diffusion Implicit Models* (DDIM), 2021
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (cosine schedule), 2021
- Lipman et al., *Flow Matching for Generative Modeling*, 2023; Liu et al., *Flow Straight and Fast* (rectified flow), 2023
- Esser et al., *Scaling Rectified Flow Transformers* (SD3, logit-normal t), 2024
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE), 2021
- Shazeer, *GLU Variants Improve Transformer* (SwiGLU), 2020
