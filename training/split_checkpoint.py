"""
Split a training checkpoint (best.pt / last.pt) into two loadable PyTorch files so
each stays under GitHub's 100 MB limit, and merge them back for resuming.

    NAME_weights.pt  everything except the optimizer: model_state, ema_state, epoch,
                     configs, metrics ... — enough for eval (python eval/evaluate.py --ckpt best_weights.pt)
    NAME_optim.pt    optimizer_state only — needed together with _weights.pt to resume training

Usage (from the repo root):
    python training/split_checkpoint.py split training/checkpoints/distill_4L_384_a07
    python training/split_checkpoint.py merge training/checkpoints/distill_4L_384_a07

`split` reads best.pt and last.pt in the folder. If a checkpoint was cut up with
`split -b 70M best.pt best.pt.part_`, the parts (best.pt.part_aa, _ab, ...) are joined
first. The original files are left in place; delete them yourself once the split looks right.
`merge` rebuilds best.pt / last.pt from the two halves.
"""
import argparse
import io
import sys
from pathlib import Path

import torch

LIMIT_MB = 99
NAMES = ("best", "last")


def load_full(folder, name):
    path = folder / f"{name}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    parts = sorted(folder.glob(f"{name}.pt.part_*"))
    if parts:
        print(f"  joining {len(parts)} parts: {', '.join(p.name for p in parts)}")
        data = b"".join(p.read_bytes() for p in parts)
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return None


def save(obj, path):
    torch.save(obj, path)
    mb = path.stat().st_size / 2**20
    flag = "" if mb <= LIMIT_MB else f"  ✗ over {LIMIT_MB} MB"
    print(f"  wrote {path.name}: {mb:.1f} MB{flag}")
    return mb <= LIMIT_MB


def split(folder):
    ok = True
    for name in NAMES:
        ckpt = load_full(folder, name)
        if ckpt is None:
            print(f"{name}: no {name}.pt or {name}.pt.part_* in {folder} — skipped")
            continue
        print(f"{name}: epoch {ckpt.get('epoch')}")
        optim   = {"epoch": ckpt.get("epoch"), "optimizer_state": ckpt.pop("optimizer_state")}
        ok &= save(ckpt, folder / f"{name}_weights.pt")
        ok &= save(optim, folder / f"{name}_optim.pt")
    if not ok:
        sys.exit(f"Some files are over {LIMIT_MB} MB")


def merge(folder):
    for name in NAMES:
        w_path, o_path = folder / f"{name}_weights.pt", folder / f"{name}_optim.pt"
        if not (w_path.exists() and o_path.exists()):
            print(f"{name}: need both {w_path.name} and {o_path.name} — skipped")
            continue
        ckpt  = torch.load(w_path, map_location="cpu", weights_only=False)
        optim = torch.load(o_path, map_location="cpu", weights_only=False)
        if optim["epoch"] != ckpt.get("epoch"):
            sys.exit(f"{name}: epoch mismatch ({w_path.name}: {ckpt.get('epoch')}, "
                     f"{o_path.name}: {optim['epoch']}) — halves are from different saves")
        ckpt["optimizer_state"] = optim["optimizer_state"]
        out = folder / f"{name}.pt"
        if out.exists():
            sys.exit(f"{out} already exists — move it away first")
        torch.save(ckpt, out)
        print(f"{name}: wrote {out.name} (epoch {ckpt.get('epoch')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split / merge checkpoints for GitHub's file limit")
    parser.add_argument("action", choices=["split", "merge"])
    parser.add_argument("folder", help="run folder with best.pt / last.pt")
    args = parser.parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    split(folder) if args.action == "split" else merge(folder)
