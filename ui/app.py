"""
AviDiff Prediction Explorer — Flask backend
Run: python ui/app.py  (from project root)
Env: NC_PATH=<path to .nc file>  (default: D:\\trajectories_adsblol_seq86_stage2.nc)
"""

import sys
import os
import re
import time
import logging
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, render_template

# Ensure project root on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.dit import TrajectoryDiT
from models.ddim import make_cosine_schedule
from models.cfm import euler_sample as cfm_euler_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ── Config ─────────────────────────────────────────────────────────────────────
NC_PATH = os.environ.get("NC_PATH", str(ROOT / "trajectories_adsblol_seq86_stage2.nc"))
MAX_FLIGHTS = int(os.environ.get("MAX_FLIGHTS", "5000"))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REGISTRY = {
    "ddim": {
        "name":  "DDIM",
        "ckpt":  str(ROOT / "checkpoints" / "best.pt"),
        "type":  "ddim",
        "arch":  "dit",
    },
    "cfm": {
        "name":  "CFM",
        "ckpt":  str(ROOT / "checkpoints_cfm" / "last.pt"),
        "type":  "cfm",
        "arch":  "dit",
    },
    "cfm_rope_og": {
        "name":  "CFM RoPE OG",
        "ckpt":  str(ROOT / "checkpoints_cfm_rope_og" / "best.pt"),
        "type":  "cfm",
        "arch":  "rope_og",
    },
    "cfm_rope_ts": {
        "name":  "CFM RoPE Timestamps",
        "ckpt":  str(ROOT / "checkpoints_cfm_rope_ts" / "best.pt"),
        "type":  "cfm",
        "arch":  "rope_ts",
    },
    "cfm_swi_dit": {
        "name":  "CFMSwiDiT",
        "ckpt":  str(ROOT / "checkpoints" / "swi_dit" / "best_swi_dit.pt"),
        "type":  "cfm",
        "arch":  "swi_dit",
    },
    "cfm_swi_dit_distill_4L": {
        "name":     "CFMSwiDiT Distill 4L",
        "ckpt":     str(ROOT / "training" / "checkpoints" / "swi_dit_distill_4L" / "best.pt"),
        "type":     "cfm",
        "arch":     "swi_dit",
        "n_layers": 4,
    },
}

# ── Global state ───────────────────────────────────────────────────────────────
_data:   dict | None = None   # loaded NetCDF data
_models: dict        = {}     # cached loaded models
_alphas: torch.Tensor | None = None  # DDIM noise schedule (cached)


# ── Classification ─────────────────────────────────────────────────────────────
# Commercial: 2-3 letter ICAO airline code + alphanumeric flight number (must contain a digit)
# e.g. UAL123, AFR83K, BAW6B, TAP237M
# GA: N-registrations (N + digit), blank callsigns, or anything else
_COMMERCIAL_RE = re.compile(r'^[A-Z]{2,3}(?=[A-Z0-9]*\d)[A-Z0-9]+$')
_GA_N_RE       = re.compile(r'^N\d')   # US registration: N followed by digit


def classify_flight(callsign: str) -> str:
    cs = (callsign or "").strip().upper()
    if not cs:
        return "ga"
    # N-number registrations are always GA regardless of anything else
    if _GA_N_RE.match(cs):
        return "ga"
    return "consumer" if _COMMERCIAL_RE.match(cs) else "ga"


# ── Coordinate helpers ──────────────────────────────────────────────────────────
_R       = 6_371_000.0
_LAT_REF = np.radians(37.6213)
_LON_REF = np.radians(-122.3790)


def xy_to_latlon(x: float, y: float):
    lat = np.degrees(y / _R + _LAT_REF)
    lon = np.degrees(x / (_R * np.cos(_LAT_REF)) + _LON_REF)
    return float(lat), float(lon)


def path_to_latlon(xy: np.ndarray) -> list[list[float]]:
    """xy: (T, 2) → [[lat, lon], ...]"""
    return [[*xy_to_latlon(float(x), float(y))] for x, y in xy]



# ── Data loading ───────────────────────────────────────────────────────────────
def _decode_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    s = str(v).strip()
    # netCDF4 masked scalars stringify as '--'
    return "" if s in ("--", "nan", "None") else s


def load_data() -> None:
    global _data
    nc_path = Path(NC_PATH)
    if not nc_path.exists():
        log.warning("NetCDF not found at %s — UI will show empty flight list", nc_path)
        _data = None
        return

    log.info("Loading NetCDF from %s …", nc_path)
    import netCDF4 as nc_lib
    ds = nc_lib.Dataset(str(nc_path), "r")

    N = min(len(ds["trajectory"]), MAX_FLIGHTS)
    # Trajectory stores ABSOLUTE x,y,z,vx,vy,vz positions in meters from SFO
    traj_raw   = ds["trajectory"][:N].astype(np.float32)   # (N, 86, 6)
    timestamps = ds["timestamps"][:N].astype(np.float64)   # (N, 86)
    icao24_raw   = [_decode_str(ds["icao24"][i])   for i in range(N)]
    callsign_raw = [_decode_str(ds["callsign"][i]) for i in range(N)]

    feat_mean  = np.array(ds.feature_mean, dtype=np.float32)  # (6,)
    feat_std   = np.array(ds.feature_std,  dtype=np.float32)  # (6,)
    t_rel_mean = float(ds.t_rel_mean)
    t_rel_std  = float(ds.t_rel_std)
    nc_offset  = int(getattr(ds, "offset", 1))
    ds.close()

    categories  = [classify_flight(cs) for cs in callsign_raw]

    flight_list = [
        {
            "id":       i,
            "icao24":   icao24_raw[i],
            "callsign": callsign_raw[i],
            "category": categories[i],
            "label":    callsign_raw[i] if callsign_raw[i] else icao24_raw[i],
        }
        for i in range(N)
    ]

    _data = {
        "traj_raw":    traj_raw,      # (N, 86, 6) absolute x,y,z,vx,vy,vz, NOT normalized
        "timestamps":  timestamps,
        "icao24":      icao24_raw,
        "callsign":    callsign_raw,
        "feat_mean":   feat_mean,
        "feat_std":    feat_std,
        "t_rel_mean":  t_rel_mean,
        "t_rel_std":   t_rel_std,
        "offset":      nc_offset,
        "categories":  categories,
        "flight_list": flight_list,
        "n": N,
    }
    n_consumer = sum(1 for c in categories if c == "consumer")
    n_ga       = N - n_consumer
    log.info("Loaded %d flights  (consumer=%d  GA=%d)", N, n_consumer, n_ga)


# ── Model loading ──────────────────────────────────────────────────────────────
def _build_model(arch: str, n_layers: int = 6) -> torch.nn.Module:
    if arch == "dit":
        return TrajectoryDiT(d_model=256, n_heads=8, n_layers=6)
    if arch == "rope_og":
        from models.dit_RoPE_original import TrajectoryDiT as RoPEDiT
        return RoPEDiT(d_model=256, n_heads=8, n_layers=6)
    if arch == "rope_ts":
        from models.dit_RoPE_timestamps import TrajectoryDiT as RoPETSDiT
        return RoPETSDiT(d_model=256, n_heads=8, n_layers=6)
    if arch == "swi_dit":
        from models.SWI_DiT import TrajectoryDiT as SwiDiT
        return SwiDiT(d_model=256, n_heads=8, n_layers=n_layers)
    raise ValueError(f"Unknown arch: {arch}")


def load_model(model_id: str) -> dict | None:
    if model_id in _models:
        return _models[model_id]

    cfg = MODEL_REGISTRY.get(model_id)
    if not cfg:
        return None
    ckpt_path = Path(cfg["ckpt"])
    if not ckpt_path.exists():
        log.warning("Checkpoint not found: %s", ckpt_path)
        return None

    log.info("Loading model %s from %s …", model_id, ckpt_path)
    try:
        model = _build_model(cfg["arch"], cfg.get("n_layers", 6)).to(DEVICE)
        ckpt  = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        state = ckpt.get("ema_state") or ckpt.get("model_state")
        model.load_state_dict(state)
        model.eval()
        _models[model_id] = {
            "model":   model,
            "type":    cfg["type"],
            "arch":    cfg["arch"],
            "epoch":   ckpt.get("epoch"),
            "val_fde": ckpt.get("val_fde"),
        }
        log.info("Loaded %s  (epoch=%s  val_fde=%s)", model_id, ckpt.get("epoch"), ckpt.get("val_fde"))
        return _models[model_id]
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return None


# ── Sampling ───────────────────────────────────────────────────────────────────
def _get_alphas() -> torch.Tensor:
    global _alphas
    if _alphas is None:
        _, _, _alphas = make_cosine_schedule(T=1000)
    return _alphas.to(DEVICE)


def ddim_sample(model, obs_t, t_rel_t, n_samples: int, n_steps: int = 50):
    """DDIM sampling — replicates visualization.py logic."""
    B  = obs_t.shape[0]  # should be 1
    ac = _get_alphas()
    T  = len(ac)

    obs_rep  = obs_t.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, 43, 6)
    trel_rep = t_rel_t.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * B, 86)

    x         = torch.randn(n_samples * B, 43, 6, device=DEVICE)
    step_size = T // n_steps
    timesteps = list(range(0, T, step_size))[::-1]

    with torch.no_grad():
        for i, t_val in enumerate(timesteps):
            t_tensor   = torch.full((n_samples * B,), t_val, device=DEVICE, dtype=torch.long)
            noise_pred = model(obs_rep, x, t_tensor.float() / T, trel_rep)
            a_bar      = ac[t_val]
            a_bar_prev = ac[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=DEVICE)
            x0_pred    = (x - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
            x          = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1 - a_bar_prev) * noise_pred

    return x.reshape(n_samples, B, 43, 6)  # (K, 1, 43, 6)


def cfm_sample(model, obs_t, t_rel_t, n_samples: int, n_steps: int = 20):
    """CFM Euler sampling."""
    return cfm_euler_sample(model, obs_t, t_rel_t, n_samples=n_samples,
                            n_steps=n_steps, device=str(DEVICE))


# ── API routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/status")
def api_status():
    return jsonify({
        "nc_loaded": _data is not None,
        "nc_path":   NC_PATH,
        "n_flights": _data["n"] if _data else 0,
        "device":    str(DEVICE),
    })


@app.route("/api/flights")
def api_flights():
    if _data is None:
        return jsonify({"ga": [], "consumer": [], "error": f"NetCDF not loaded ({NC_PATH})"}), 200
    fl = _data["flight_list"]
    return jsonify({
        "ga":       [f for f in fl if f["category"] == "ga"],
        "consumer": [f for f in fl if f["category"] == "consumer"],
    })


@app.route("/api/models")
def api_models():
    result = []
    for mid, cfg in MODEL_REGISTRY.items():
        ckpt_exists = Path(cfg["ckpt"]).exists()
        loaded      = _models.get(mid)
        result.append({
            "id":        mid,
            "name":      cfg["name"],
            "available": ckpt_exists,
            "loaded":    mid in _models,
            "epoch":     loaded["epoch"]   if loaded else None,
            "val_fde":   float(loaded["val_fde"]) if loaded and loaded["val_fde"] is not None else None,
        })
    return jsonify(result)


def _extended_gt(flight_idx: int, total_steps: int) -> np.ndarray:
    """
    Build extended ground truth beyond 43 steps by chaining consecutive windows.

    The NC file uses offset=10, so window (base+k) starts 10 pings after window (base+k-1).
    Window base+k's last OFFSET=10 steps are the 10 NEW future points not in window base+k-1.

    Continuity check: ts[base+k][0] should equal ts[base+k-1][OFFSET].
    Max GT = 43 + n_chained_windows * OFFSET steps.
    """
    traj     = _data["traj_raw"]
    ts       = _data["timestamps"]
    icao_arr = _data["icao24"]
    N        = _data["n"]
    OFFSET   = _data.get("offset", 10)   # window stride in pings

    base_icao = icao_arr[flight_idx]

    # Start: first 43 GT steps from current window
    gt = [traj[flight_idx][43:]]   # [(43, 6)]
    steps = 43

    # Rolling expected start timestamp of the next window
    expected_next_start = float(ts[flight_idx][OFFSET])

    k = 1
    while steps < total_steps:
        nxt = flight_idx + k
        if nxt >= N or icao_arr[nxt] != base_icao:
            break
        if abs(ts[nxt][0] - expected_next_start) > 10.0:
            break

        # This window's last OFFSET steps are the new future points
        new_block = traj[nxt][-OFFSET:]         # (OFFSET, 6)
        remaining = total_steps - steps
        gt.append(new_block[:remaining])
        steps += min(OFFSET, remaining)

        expected_next_start = float(ts[nxt][OFFSET])
        k += 1

    return np.concatenate(gt, axis=0)[:total_steps]  # (T, 6)


def _kde_nll(samples_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """Negative log-likelihood of gt_xy under a Gaussian KDE fit to samples_xy (K, 2).

    Isotropic bandwidth via Scott's rule for d=2. Returns nats.
    """
    K = samples_xy.shape[0]
    if K < 2:
        return float("nan")
    std = float(samples_xy.std(axis=0).mean())
    h = max(std * K ** (-1.0 / 6.0), 1e-3)
    sqdist     = np.sum((gt_xy[None, :] - samples_xy) ** 2, axis=1)  # (K,)
    log_kernel = -sqdist / (2 * h ** 2) - np.log(2 * np.pi * h ** 2)
    log_density = np.logaddexp.reduce(log_kernel) - np.log(K)
    return float(-log_density)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if _data is None:
        return jsonify({"error": "Data not loaded"}), 503

    body           = request.get_json(force=True)
    flight_idx     = int(body["flight_idx"])
    model_id       = body.get("model", "ddim")
    n_samples      = max(1,  min(int(body.get("n_samples",    10)),  200))
    total_horizon  = max(1,  min(int(body.get("total_horizon", 43)), 500))
    obs_window     = max(1,  min(int(body.get("obs_window",   43)),   43))
    n_steps        = max(10, min(int(body.get("n_steps",      50)),  200))

    if not (0 <= flight_idx < _data["n"]):
        return jsonify({"error": "Invalid flight_idx"}), 400

    mi = load_model(model_id)
    if mi is None:
        return jsonify({"error": f"Model '{model_id}' not available — checkpoint missing"}), 400

    model      = mi["model"]
    model_type = mi["type"]

    # ── Raw trajectory (absolute x,y,z,vx,vy,vz in metres from SFO) ──────────
    traj_raw  = _data["traj_raw"][flight_idx]   # (86, 6)
    ts        = _data["timestamps"][flight_idx]  # (86,)
    feat_mean = _data["feat_mean"]               # (6,)
    feat_std  = _data["feat_std"]                # (6,)

    obs_raw = traj_raw[:43]   # (43, 6)
    fut_raw = traj_raw[43:]   # (43, 6) — ground truth, available for steps 1-43 only

    # ── Normalize ────────────────────────────────────────────────────────────
    obs_norm_full = (obs_raw - feat_mean) / (feat_std + 1e-8)  # (43, 6)
    t_rel         = ts - ts[0]
    t_rel_norm    = (t_rel - _data["t_rel_mean"]) / (_data["t_rel_std"] + 1e-8)  # (86,)

    # ── Apply obs_window: pad earlier steps by repeating the first visible pt ─
    if obs_window < 43:
        start = 43 - obs_window
        obs_norm_windowed = np.tile(obs_norm_full[start], (43, 1)).copy()
        obs_norm_windowed[start:] = obs_norm_full[start:]
    else:
        obs_norm_windowed = obs_norm_full

    obs_t   = torch.tensor(obs_norm_windowed, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,43,6)
    t_rel_t = torch.tensor(t_rel_norm,        dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,86)

    def _sample_block(obs_tensor, trel_tensor):
        """Run one 43-step sampling block. Returns (K, 43, 6) normalised."""
        if model_type == "ddim":
            out = ddim_sample(model, obs_tensor, trel_tensor, n_samples if obs_tensor.shape[0] == 1 else 1, n_steps)
        else:
            out = cfm_sample(model, obs_tensor, trel_tensor, n_samples if obs_tensor.shape[0] == 1 else 1)
        # out: (n_samples, B, 43, 6) — squeeze/reshape to (K, 43, 6)
        K = out.shape[0] * out.shape[1]
        return out.reshape(K, 43, 6)

    # ── Autoregressive rollout ────────────────────────────────────────────────
    t0 = time.perf_counter()

    n_blocks = int(np.ceil(total_horizon / 43))   # how many 43-step blocks needed
    all_preds_raw = []                             # list of (K, 43, 6) raw arrays

    current_obs_t   = obs_t      # (1, 43, 6) or (K, 43, 6) for later blocks
    current_trel_t  = t_rel_t    # (1, 86)   reused for every block

    for block in range(n_blocks):
        preds_norm = _sample_block(current_obs_t, current_trel_t)  # (K, 43, 6) normalised
        preds_raw_block = preds_norm.cpu().numpy() * feat_std + feat_mean  # (K, 43, 6)
        all_preds_raw.append(preds_raw_block)

        if block < n_blocks - 1:
            # Use this block's predictions as obs for the next block
            # Re-normalise and tile t_rel for all K samples
            next_obs_norm = (preds_raw_block - feat_mean) / (feat_std + 1e-8)   # (K, 43, 6)
            current_obs_t  = torch.tensor(next_obs_norm, dtype=torch.float32, device=DEVICE)
            current_trel_t = t_rel_t.expand(n_samples, -1)   # (K, 86) reuse original timing

    elapsed = time.perf_counter() - t0

    # Concatenate blocks and trim to total_horizon
    preds_raw_full = np.concatenate(all_preds_raw, axis=1)[:, :total_horizon, :]  # (K, H, 6)

    # ── Extended ground truth (chain consecutive windows) ─────────────────────
    gt_full = _extended_gt(flight_idx, total_horizon)   # (T, 6)  T <= total_horizon
    gt_steps_available = len(gt_full)                   # how many GT steps we got

    # ── FDE/ADE at min(total_horizon, available GT) ─────────────────────────────
    fde_horizon = min(total_horizon, gt_steps_available)
    gt_end      = gt_full[fde_horizon - 1, :2]
    pred_end    = preds_raw_full[:, fde_horizon - 1, :2]
    fde_each    = np.linalg.norm(pred_end - gt_end, axis=1)

    # ADE: mean per-step displacement over the same horizon, one value per sample
    gt_path    = gt_full[:fde_horizon, :2]                          # (H, 2)
    pred_path  = preds_raw_full[:, :fde_horizon, :2]                # (K, H, 2)
    step_disp  = np.linalg.norm(pred_path - gt_path[None, :, :], axis=2)  # (K, H)
    ade_each   = step_disp.mean(axis=1)                             # (K,)

    # KDE-NLL: fit a per-step Gaussian KDE over the K samples, evaluate at GT,
    # average -log(density) over the horizon (nats; lower = better calibrated).
    kde_nll_per_step = np.array([
        _kde_nll(pred_path[:, s, :], gt_path[s, :]) for s in range(fde_horizon)
    ])
    kde_nll_mean = float(np.nanmean(kde_nll_per_step))

    # Display obs using only the visible window portion
    obs_display = obs_raw[43 - obs_window:] if obs_window < 43 else obs_raw

    return jsonify({
        "obs":          path_to_latlon(obs_display[:, :2]),
        "future":       path_to_latlon(gt_full[:, :2]),          # extended GT up to total_horizon
        "predictions":  [path_to_latlon(preds_raw_full[k, :, :2]) for k in range(n_samples)],
        "fde": {
            "min":        float(fde_each.min()),
            "mean":       float(fde_each.mean()),
            "max":        float(fde_each.max()),
            "per_sample": fde_each.tolist(),
            "at_step":    fde_horizon,
            "has_gt":     fde_horizon == total_horizon,
        },
        "ade": {
            "min":        float(ade_each.min()),
            "mean":       float(ade_each.mean()),
            "max":        float(ade_each.max()),
            "per_sample": ade_each.tolist(),
            "at_step":    fde_horizon,
            "has_gt":     fde_horizon == total_horizon,
        },
        "kde_nll": {
            "mean":       kde_nll_mean,
            "at_step":    fde_horizon,
            "has_gt":     fde_horizon == total_horizon,
        },
        "total_horizon":       total_horizon,
        "gt_steps_available":  gt_steps_available,
        "obs_window":          obs_window,
        "n_samples":           n_samples,
        "elapsed_s":           elapsed,
        "n_blocks":            n_blocks,
        "model_id":            model_id,
        "flight_idx":          flight_idx,
        "callsign":            _data["callsign"][flight_idx],
        "icao24":              _data["icao24"][flight_idx],
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AviDiff Prediction Explorer")
    parser.add_argument("--nc",   default=NC_PATH, help="Path to .nc file")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-flights", default=MAX_FLIGHTS, type=int)
    args = parser.parse_args()

    NC_PATH     = args.nc
    MAX_FLIGHTS = args.max_flights

    load_data()
    log.info("Pre-loading DDIM model…")
    load_model("ddim")
    log.info("Starting server on http://%s:%d", args.host, args.port)
    app.run(debug=False, port=args.port, host=args.host)
