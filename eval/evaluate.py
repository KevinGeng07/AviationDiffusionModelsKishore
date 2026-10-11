"""
Evaluate finished CFM SWI_DiT checkpoints on the test set.

Usage (from the repo root):
    python eval/evaluate.py                                   # every folder in FOLDERS below
    python eval/evaluate.py finished_ckpt/6_2_128_SWiDiT      # or folders given on the command line
    python eval/evaluate.py --n_samples 20 --max_batches 10

Each folder can be the run folder itself or any parent of it — the script finds
best.pt (or --ckpt last.pt) and student_config.json inside it.

For every test window the model samples K possible futures (K = --n_samples).
Errors are distances in metres between predicted and true (x, y) positions:

    ADE  = average error over all 43 future steps     (how good is the whole path)
    FDE  = error at the last future step only           (how good is the endpoint)
    min  = error of the best of the K samples           (did ANY sample get it right)
    avg  = error averaged over the K samples            (how good is a typical sample)

So you get minADE, minFDE, avgADE, avgFDE, each averaged over all test windows.
"""
import argparse
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DataLoaders.ADSBdataset import get_dataloaders
from models.SWI_DiT import TrajectoryDiT
from models.cfm import euler_sample

# Folders to evaluate when none are given on the command line — add or remove lines.
# Paths are relative to the repo root (absolute paths work too).
FOLDERS = [
    "finished_ckpt/6_2_128_SWiDiT",
    "finished_ckpt/6_2_256_SWiDiT",
    "finished_ckpt/6_2_384_SWiDiT",
    "finished_ckpt/6_4_128_SWiDiT",
    "finished_ckpt/6_4_256_SWiDiT",
    "finished_ckpt/6_5_SWiDiT",
]


def find_file(folder, name):
    matches = sorted(Path(folder).rglob(name))
    return matches[0] if matches else None


def load_model(folder, ckpt_name, device):
    ckpt_path = find_file(folder, ckpt_name)
    if ckpt_path is None:
        sys.exit(f"No {ckpt_name} found under {folder}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # model size: student_config.json next to the checkpoint, else the copy saved inside it
    cfg_path = ckpt_path.parent / "student_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    elif "student_config" in ckpt:
        cfg = ckpt["student_config"]
    else:
        sys.exit(f"No student_config.json next to {ckpt_path} and none stored in the checkpoint")

    model = TrajectoryDiT(**cfg["model"]).to(device)
    model.load_state_dict(ckpt.get("ema_state") or ckpt["model_state"])   # EMA weights, like training's val
    model.eval()

    seed = ckpt.get("train_config", {}).get("seed", 42)
    print(f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})")
    print(f"Model: {cfg['model']} | {sum(p.numel() for p in model.parameters()):,} params")
    return model, seed


def evaluate(model, test_loader, feat_std, device, n_samples, n_steps, max_batches):
    std_xy = torch.tensor(feat_std[:2], device=device, dtype=torch.float32)
    totals = {"minADE": 0.0, "minFDE": 0.0, "avgADE": 0.0, "avgFDE": 0.0}
    n_windows = 0

    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            if max_batches is not None and b >= max_batches:
                break
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            preds = euler_sample(model, obs, t_rel, n_samples=n_samples,
                                 n_steps=n_steps, device=str(device))   # (K, B, 43, 6)

            # back to metres: only x, y; the mean cancels in a difference, so multiply by std
            pred_xy = preds[..., :2] * std_xy            # (K, B, 43, 2)
            true_xy = fut[..., :2] * std_xy              # (B, 43, 2)

            err = torch.norm(pred_xy - true_xy[None], dim=-1)   # (K, B, 43) metres per step
            ade = err.mean(dim=-1)                              # (K, B) average over steps
            fde = err[..., -1]                                  # (K, B) last step

            totals["minADE"] += ade.min(dim=0).values.sum().item()
            totals["minFDE"] += fde.min(dim=0).values.sum().item()
            totals["avgADE"] += ade.mean(dim=0).sum().item()
            totals["avgFDE"] += fde.mean(dim=0).sum().item()
            n_windows += obs.shape[0]
            print(f"  batch {b + 1}/{len(test_loader)} | running minFDE "
                  f"{totals['minFDE'] / n_windows:.1f}m", end="\r", flush=True)

    print()
    return {k: v / n_windows for k, v in totals.items()}, n_windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-set ADE/FDE for finished checkpoints")
    parser.add_argument("folders", nargs="*", help="run folders (default: FOLDERS in this file)")
    parser.add_argument("--nc_path", default=str(ROOT / "Data" / "trajectories_adsblol_seq86_stage2.nc"))
    parser.add_argument("--ckpt", default="best.pt", help="best.pt or last.pt")
    parser.add_argument("--n_samples", type=int, default=20, help="K futures sampled per window")
    parser.add_argument("--n_steps", type=int, default=20, help="Euler steps (training used 20)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=None, help="limit for a quick check")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    folders = args.folders or FOLDERS
    folders = [f if Path(f).is_absolute() or Path(f).exists() else ROOT / f for f in folders]
    missing = [str(f) for f in folders if not Path(f).is_dir()]
    if missing:
        sys.exit("Folder(s) not found: " + ", ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = nc.Dataset(args.nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()

    test_loaders = {}   # one per training seed; the seed decides the aircraft-level split
    all_results = {}
    for folder in folders:
        print(f"\n=== {folder} ===")
        model, seed = load_model(folder, args.ckpt, device)
        if seed not in test_loaders:
            # same seed as training → same split, so these aircraft were never trained on
            _, _, test_loaders[seed] = get_dataloaders(args.nc_path, batch_size=args.batch_size,
                                                       seed=seed, num_workers=args.num_workers)

        torch.manual_seed(0)   # same sampling noise for every model → comparable numbers
        results, n = evaluate(model, test_loaders[seed], feat_std, device,
                              args.n_samples, args.n_steps, args.max_batches)
        all_results[Path(folder).name] = results
        for name, value in results.items():
            print(f"  {name}: {value:8.1f} m")

    # side-by-side summary
    print(f"\nTest set: {n:,} windows | K={args.n_samples} samples | {args.n_steps} Euler steps | metres")
    names = list(next(iter(all_results.values())))
    width = max(len(k) for k in all_results) + 2
    print("model".ljust(width) + "".join(f"{m:>10}" for m in names))
    for model_name, results in all_results.items():
        print(model_name.ljust(width) + "".join(f"{results[m]:10.1f}" for m in names))
