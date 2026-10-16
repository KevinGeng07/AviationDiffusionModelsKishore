"""
Progressive distillation (Salimans & Ho, 2022) for CFM + SWI_DiT: fewer sampling steps.

The architecture stays the same; what shrinks is the number of Euler steps the
sampler needs. Starting from any trained model (--base, default 6_4_256), each
round trains a student to cover in ONE Euler step what the frozen teacher
covers in TWO, halving the step count:

    round 1:  teacher @ 32 steps → student @ 16 steps     (teacher = base model)
    round 2:  teacher @ 16 steps → student @  8 steps     (teacher = round-1 student)
    ...       down to --min_steps (default 1)

The student of every round is initialized from its teacher's weights (so the
two share an architecture), and the next round's teacher is this round's best
EMA student.

Per-round loss, on the student's time grid t ∈ {0, 1/N, ..., (N-1)/N}
(N = student steps, dt = 1/(2N) = teacher step):

    x_t      = (1 - t)·noise + t·fut               (same interpolation as CFM training)
    v1       = teacher(obs, x_t,         t,      t_rel)
    v2       = teacher(obs, x_t + v1·dt, t + dt, t_rel)
    v_tgt    = (v1 + v2) / 2                       (= (x_{t+2dt} - x_t) / 2dt)
    L        = weighted_mse(student(obs, x_t, t, t_rel), v_tgt)

so one student Euler step of size 2dt lands exactly where two teacher steps do.

Selecting a model (--base, and --teacher for an optional different round-1 teacher):

    6_4_256                            shorthand for finished_ckpt/6_4_256_SWiDiT
    teacher                            the original 6-layer model (configs/swi_dit_teacher.json)
    finished_ckpt/6_4_384_SWiDiT       any run folder; best.pt (--ckpt_name) is found inside,
                                       split checkpoints (best_0.pt, ...) are joined
    path/to/some.pt                    a checkpoint file

The architecture comes from --base_config / --teacher_config if given, else
student_config.json next to the checkpoint, else the config stored inside it.

Output layout (each round folder works with eval/evaluate.py --n_steps N):

    <output_dir>/steps_16/{best.pt, last.pt, metrics.csv, student_config.json, done}
    <output_dir>/steps_08/...

Resume: finished rounds (with a `done` file) are skipped and their best.pt is
loaded as the next teacher; an unfinished round resumes from its last.pt.
"""

import argparse
import csv
import json
import os
import re
import time
import torch
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DataLoaders.ADSBdataset import get_dataloaders
from models.SWI_DiT import TrajectoryDiT
from models.cfm import forward_cfm, euler_sample
from eval.evaluate import load_checkpoint
from training.train_cfm_swi_dit_distill_savio import (
    EMA, set_seed, resolve_path, load_config, weighted_mse,
)

CHECKPOINT_DIR = ROOT / "training" / "checkpoints"
TEACHER_CONFIG = "configs/swi_dit_teacher.json"
# Where the original teacher lives locally and on Savio
TEACHER_CKPTS  = ["checkpoints/swi_dit/best_swi_dit.pt",
                  "training/checkpoints/swi_dit/best_swi_dit.pt"]

METRIC_FIELDS = [
    "epoch", "lr", "epoch_time_s", "train_loss", "val_loss", "val_minfde",
]


# ── Model selection ───────────────────────────────────────────────────────

def run_label(spec):
    """Short name for a --base spec, used in the default output folder."""
    if spec == "teacher":
        return "6_6_256"
    name = Path(spec).name
    if name.endswith(".pt"):
        name = Path(spec).parent.name
    return name.removesuffix("_SWiDiT")


def load_model_spec(spec, config_path, ckpt_name, device):
    """
    Resolve a model spec (see module docstring) to a frozen TrajectoryDiT with EMA
    weights. Returns (model, config dict, checkpoint path).
    """
    if spec == "teacher":
        config_path = config_path or TEACHER_CONFIG
        candidates  = [load_config(config_path).get("checkpoint")] + TEACHER_CKPTS
        found       = [resolve_path(p) for p in candidates if p and resolve_path(p).exists()]
        if not found:
            sys.exit(f"Original teacher checkpoint not found (tried {candidates})")
        ckpt_path = found[0]
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        if re.fullmatch(r"\d+_\d+(_\d+)?", spec):           # e.g. 6_4_256
            spec = f"finished_ckpt/{spec}_SWiDiT"
        path = resolve_path(spec)
        if path.is_file():
            ckpt_path = path
            ckpt      = torch.load(path, map_location=device, weights_only=False)
        elif path.is_dir():
            ckpt, ckpt_path = load_checkpoint(path, ckpt_name, device)
        else:
            sys.exit(f"Model not found: {spec}")

    # Architecture: explicit config > student_config.json beside it > copy in the checkpoint
    if config_path:
        cfg = load_config(config_path)
    elif (ckpt_path.parent / "student_config.json").exists():
        cfg = load_config(ckpt_path.parent / "student_config.json")
    elif ckpt.get("student_config"):
        cfg = ckpt["student_config"]
    else:
        sys.exit(f"No architecture for {ckpt_path}: pass a config JSON")

    model = TrajectoryDiT(**cfg["model"]).to(device)
    model.load_state_dict(ckpt.get("ema_state") or ckpt["model_state"])
    model.eval()
    model.requires_grad_(False)
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch')}, val_fde {ckpt.get('val_fde')}) | "
          f"{cfg['model']} | {sum(p.numel() for p in model.parameters()):,} params")
    return model, cfg, ckpt_path


def frozen_copy(cfg, state, device):
    model = TrajectoryDiT(**cfg["model"]).to(device)
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    return model


# ── Distillation pieces ───────────────────────────────────────────────────

def round_schedule(start_steps, min_steps):
    """
    (teacher steps, student steps) per round, e.g. 32 → [(32, 16), (16, 8), ..., (2, 1)].
    Stops at an odd count (20 → 10 → 5), since a student step must cover exactly two teacher steps.
    """
    rounds = []
    steps  = start_steps
    while steps % 2 == 0 and steps // 2 >= min_steps:
        rounds.append((steps, steps // 2))
        steps //= 2
    return rounds


def two_step_target(teacher, obs, x_t, t, t_rel, dt):
    """Velocity that moves x_t to where two teacher Euler steps of size dt end up."""
    v1   = teacher(obs, x_t, t, t_rel)
    x_md = x_t + v1 * dt
    v2   = teacher(obs, x_md, t + dt, t_rel)
    return 0.5 * (v1 + v2)


def sample_grid_time(B, n_steps, device):
    """t uniformly from the student's Euler grid {0, 1/N, ..., (N-1)/N}."""
    k = torch.randint(0, n_steps, (B,), device=device)
    return k.float() / n_steps


def validate_fde(model, val_loader, device, feat_std, seed, n_steps, max_batches=20):
    """minFDE@5 in metres with an n_steps Euler sampler."""
    model.eval()
    torch.manual_seed(seed)
    feat_std_xy = torch.tensor(feat_std[:2], device=device, dtype=torch.float32)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= max_batches:
                break
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)
            preds = euler_sample(model, obs, t_rel, n_samples=5,
                                 n_steps=n_steps, device=str(device))
            fde   = torch.norm(preds[:, :, -1, :2] * feat_std_xy
                               - fut[None, :, -1, :2] * feat_std_xy, dim=-1)
            total += fde.min(dim=0).values.mean().item()
            n     += 1
    return total / n


def validate_loss(student, teacher, val_loader, device, seed, n_steps, max_batches=20):
    student.eval()
    torch.manual_seed(seed)
    dt = 0.5 / n_steps
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= max_batches:
                break
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)
            t         = sample_grid_time(obs.shape[0], n_steps, device)
            x_t, _, _ = forward_cfm(fut, t)
            v_tgt     = two_step_target(teacher, obs, x_t, t, t_rel, dt)
            total    += weighted_mse(student(obs, x_t, t, t_rel), v_tgt).item()
            n        += 1
    return total / n


def append_metrics(path, row):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def train_round(teacher, student_cfg, init_state, teacher_steps, student_steps,
                train_loader, val_loader, feat_std, round_dir, device, args, run_info):
    """One halving round. Returns the path of this round's best checkpoint."""
    round_dir.mkdir(parents=True, exist_ok=True)
    with open(round_dir / "student_config.json", "w") as f:
        json.dump(student_cfg, f, indent=2)
    metrics_path = round_dir / "metrics.csv"

    model = TrajectoryDiT(**student_cfg["model"]).to(device)
    model.load_state_dict(init_state)              # student starts as a copy of the teacher
    ema   = EMA(model, decay=args.ema_decay)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # cosine over the round, decaying to min_lr_frac · lr instead of 0. It steps at the end
    # of each epoch, so T_max = epochs - 1 makes the last epoch train exactly at the floor.
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_per_round - 1),
                                  eta_min=args.lr * args.min_lr_frac)
    dt        = 0.5 / student_steps                # teacher step size

    start_epoch, global_step, best_fde = 0, 0, float("inf")
    last_ckpt = round_dir / "last.pt"
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_fde    = ckpt["best_fde"]
        print(f"  resumed round at epoch {start_epoch}")

    # References: teacher at its own step count, and the undistilled init at the student's
    teacher_fde = validate_fde(teacher, val_loader, device, feat_std, args.seed, teacher_steps)
    init_model  = frozen_copy(student_cfg, init_state, device)
    init_fde    = validate_fde(init_model, val_loader, device, feat_std, args.seed, student_steps)
    del init_model
    print(f"  teacher @ {teacher_steps} steps: {teacher_fde:.1f}m | "
          f"student init @ {student_steps} steps (no distillation): {init_fde:.1f}m", flush=True)

    for epoch in range(start_epoch, args.epochs_per_round):
        torch.manual_seed(args.seed + 1000 * student_steps + epoch)
        epoch_start = time.time()
        epoch_lr    = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            t         = sample_grid_time(obs.shape[0], student_steps, device)
            x_t, _, _ = forward_cfm(fut, t)
            with torch.no_grad():
                v_tgt = two_step_target(teacher, obs, x_t, t, t_rel, dt)

            loss = weighted_mse(model(obs, x_t, t, t_rel), v_tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if global_step < args.warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * (global_step + 1) / args.warmup_steps

            ema.update(model)
            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if args.log_every and n_batches % args.log_every == 0:
                elapsed = time.time() - epoch_start
                print(f"    epoch {epoch+1:03d} | batch {n_batches:,}/{len(train_loader):,} | "
                      f"loss {total_loss / n_batches:.5f} | {n_batches / elapsed:.1f} it/s",
                      flush=True)

        scheduler.step()

        val_loss = validate_loss(ema.shadow, teacher, val_loader, device, args.seed, student_steps)
        val_fde  = validate_fde(ema.shadow, val_loader, device, feat_std, args.seed, student_steps)
        metrics  = {
            "epoch":        epoch + 1,
            "lr":           f"{epoch_lr:.3e}",
            "train_loss":   f"{total_loss / n_batches:.6f}",
            "val_loss":     f"{val_loss:.6f}",
            "val_minfde":   f"{val_fde:.1f}",
            "epoch_time_s": f"{time.time() - epoch_start:.1f}",
        }
        append_metrics(metrics_path, metrics)

        checkpoint = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "ema_state":       ema.shadow.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step":     global_step,
            "best_fde":        best_fde,
            "val_fde":         val_fde,
            "n_steps":         student_steps,
            "teacher_steps":   teacher_steps,
            "teacher_val_fde": teacher_fde,
            "init_val_fde":    init_fde,
            "student_config":  student_cfg,
            "train_config":    {**vars(args), **run_info},
            "metrics":         metrics,
        }
        mark = ""
        if val_fde < best_fde:
            best_fde = checkpoint["best_fde"] = val_fde
            torch.save(checkpoint, round_dir / "best.pt")
            mark = "  ✓ best"
        torch.save(checkpoint, last_ckpt)

        print(f"  [{student_steps} steps] epoch {epoch+1:03d} | train {metrics['train_loss']} | "
              f"val {metrics['val_loss']} | minFDE {val_fde:.1f}m "
              f"(teacher@{teacher_steps} {teacher_fde:.1f}m, init {init_fde:.1f}m){mark}",
              flush=True)

    (round_dir / "done").write_text(f"{best_fde:.1f}\n")
    return round_dir / "best.pt"


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir or CHECKPOINT_DIR / f"progressive_{run_label(args.base)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    train_loader, val_loader, _ = get_dataloaders(
        args.nc_path, batch_size=args.batch_size, seed=args.seed,
        subset=args.subset, num_workers=num_workers,
    )
    import netCDF4 as nc
    ds       = nc.Dataset(args.nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()

    print(f"Progressive distillation on {device} → {output_dir}")
    print("Base (student init):", end=" ")
    base_model, base_cfg, base_ckpt = load_model_spec(
        args.base, args.base_config, args.ckpt_name, device)
    init_state = base_model.state_dict()

    # Round-1 teacher: the base model unless --teacher picks another one
    if args.teacher:
        print("Round-1 teacher:", end=" ")
        teacher, _, teacher_ckpt = load_model_spec(
            args.teacher, args.teacher_config, args.ckpt_name, device)
    else:
        teacher, teacher_ckpt = base_model, base_ckpt
        print("Round-1 teacher: the base model")
    run_info = {"base_ckpt": str(base_ckpt), "round1_teacher_ckpt": str(teacher_ckpt)}

    for teacher_steps, student_steps in round_schedule(args.start_steps, args.min_steps):
        round_dir = output_dir / f"steps_{student_steps:02d}"
        print(f"\n=== Round: {teacher_steps} → {student_steps} steps ({round_dir}) ===", flush=True)

        if (round_dir / "done").exists():
            best_path = round_dir / "best.pt"
            print(f"  already done (best minFDE {(round_dir / 'done').read_text().strip()}m), skipping")
        else:
            best_path = train_round(teacher, base_cfg, init_state, teacher_steps, student_steps,
                                    train_loader, val_loader, feat_std, round_dir, device,
                                    args, run_info)

        # Best EMA student becomes the next teacher and the next student's init
        best       = torch.load(best_path, map_location=device, weights_only=False)
        teacher    = frozen_copy(base_cfg, best["ema_state"], device)
        init_state = teacher.state_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=str, required=True,
                        help="Path to the .nc trajectory dataset")
    parser.add_argument("--base", type=str, default="6_4_256",
                        help="Model whose sampler is distilled: shorthand (6_4_256), 'teacher', "
                             "a run folder or a .pt file")
    parser.add_argument("--base_config", type=str, default=None,
                        help="Architecture JSON for --base (default: found next to / inside the checkpoint)")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Optional different round-1 teacher (same spec format as --base); "
                             "default: the base model itself")
    parser.add_argument("--teacher_config", type=str, default=None,
                        help="Architecture JSON for --teacher")
    parser.add_argument("--ckpt_name", type=str, default="best.pt",
                        help="Checkpoint to use inside run folders (best.pt or last.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: training/checkpoints/progressive_<base>")
    parser.add_argument("--start_steps", type=int, default=32,
                        help="Euler steps of the round-1 teacher; halved every round")
    parser.add_argument("--min_steps", type=int, default=1,
                        help="Stop once the student reaches this many steps")
    parser.add_argument("--epochs_per_round", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak LR of every round (after warmup)")
    parser.add_argument("--min_lr_frac", type=float, default=0.2,
                        help="Floor of the per-round cosine as a fraction of --lr (0.2 → decays to 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=250,
                        help="Linear LR warmup at the start of every round")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds the train/val/test split and all RNGs (keep 42 to match the base run)")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers; default = $SLURM_CPUS_PER_TASK, or 4")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit to N samples; omit for the full dataset")
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())
