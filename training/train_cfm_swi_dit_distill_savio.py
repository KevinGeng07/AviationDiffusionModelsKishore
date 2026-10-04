"""
Knowledge distillation for CFM + SWI_DiT: teacher → smaller student.

Teacher and student architectures come from separate JSON config files
(see configs/swi_dit_teacher.json, configs/swi_dit_student.json), passed
with --teacher_config / --student_config. The teacher config also names the
trained checkpoint to load (overridable with --teacher_ckpt).

The teacher is frozen (EMA weights, eval mode). The student is trained with
the standard CFM setup (same flow-time sampling, interpolation, endpoint
weighting, optimizer, EMA, validation) plus a term matching the teacher's
velocity.

Distillation loss (both terms use the same endpoint-weighted MSE as the
standard CFM loss: last quarter of the future horizon weighted ×2):

    x_t      = (1 - t)·noise + t·fut              (shared by student and teacher)
    v_tgt    = fut - noise                        (ground-truth CFM target)
    v_s      = student(obs, x_t, t, t_rel)
    v_T      = teacher(obs, x_t, t, t_rel)        (no grad, eval mode)

    L_gt     = weighted_mse(v_s, v_tgt)           ("normal" CFM loss)
    L_kd     = weighted_mse(v_s, v_T)
    L        = alpha · L_gt + (1 - alpha) · L_kd     (default alpha = 0.5)

Per-epoch metrics (printed and appended to <output_dir>/metrics.csv), all for
the EMA student unless noted:
    train_loss, train_loss_gt, train_loss_kd   averaged over the epoch (raw student)
    val_loss, val_loss_gt, val_loss_kd         same losses on validation batches
    val_kl_endpoint, val_kl_path               KL(teacher ‖ student) between Gaussians
                                               fitted to K sampled (x, y) positions,
                                               at the final step / averaged over all
                                               future steps (nats, lower = closer)
    val_minfde_rough                           minFDE@5 on 5 val batches (metres)
    val_minfde                                 minFDE@5 on 20 val batches, every 5 epochs

Reproducibility: --seed fixes the train/val/test aircraft split (via
get_dataloaders) and Python/NumPy/PyTorch RNGs. The torch RNG is re-seeded
with (seed + epoch) at the start of every epoch, so batch order and the CFM
noise / flow times of a given epoch are the same whether or not the run was
resumed from last.pt. Validation metrics use fixed seeds, so they are
comparable across epochs.
"""

import argparse
import csv
import json
import random
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DataLoaders.ADSBdataset import get_dataloaders
from models.SWI_DiT import TrajectoryDiT
from models.cfm import sample_flow_time, forward_cfm, euler_sample

CHECKPOINT_DIR = ROOT / "training" / "checkpoints"


METRIC_FIELDS = [
    "epoch", "lr", "epoch_time_s",
    "train_loss", "train_loss_gt", "train_loss_kd",
    "val_loss", "val_loss_gt", "val_loss_kd",
    "val_kl_endpoint", "val_kl_path",
    "val_minfde_rough", "val_minfde",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path):
    """Paths in configs may be relative to the repo root (training jobs often cd elsewhere)."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def load_config(path):
    with open(resolve_path(path)) as f:
        cfg = json.load(f)
    if "model" not in cfg:
        raise ValueError(f"{path}: config needs a 'model' section of TrajectoryDiT kwargs")
    return cfg


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()

    def update(self, model):
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data = self.decay * s_param.data + (1 - self.decay) * param.data

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


def weighted_mse(pred, target):
    """Standard CFM loss weighting: last quarter of the future horizon ×2."""
    fut_len = pred.shape[1]
    weights = torch.ones(fut_len, device=pred.device)
    weights[-fut_len // 4:] = 2.0
    loss_plain = nn.functional.mse_loss(pred, target, reduction="none")
    return (loss_plain * weights[None, :, None]).mean()


def gaussian_kl(mu_p, cov_p, mu_q, cov_q):
    """KL(P ‖ Q) between multivariate Gaussians; batched over leading dims. Returns nats."""
    k       = mu_p.shape[-1]
    cov_q_i = torch.linalg.inv(cov_q)
    diff    = (mu_q - mu_p).unsqueeze(-1)
    trace   = torch.diagonal(cov_q_i @ cov_p, dim1=-2, dim2=-1).sum(-1)
    maha    = (diff.transpose(-2, -1) @ cov_q_i @ diff).squeeze(-1).squeeze(-1)
    logdet  = torch.logdet(cov_q) - torch.logdet(cov_p)
    return 0.5 * (trace + maha - k + logdet)


def fit_gaussian(samples, eps=1.0):
    """samples: (K, ..., d) → mean (..., d), covariance (..., d, d) with eps·I ridge."""
    K      = samples.shape[0]
    mu     = samples.mean(dim=0)
    c      = samples - mu
    cov    = torch.einsum("k...i,k...j->...ij", c, c) / (K - 1)
    eye    = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)
    return mu, cov + eps * eye


def validate(model, val_loader, device, feat_std, seed, max_batches=20):
    """minFDE over 5 samples, in metres. Takes a plain model (student EMA or teacher)."""
    model.eval()
    torch.manual_seed(seed)
    total_fde = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break

            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            preds = euler_sample(model, obs, t_rel,
                                  n_samples=5, n_steps=20, device=str(device))

            feat_std_xy    = torch.tensor(feat_std[:2], device=device)
            preds_xy       = preds[..., :2] * feat_std_xy
            fut_xy         = fut[..., :2]   * feat_std_xy
            fde_per_sample = torch.norm(
                preds_xy[:, :, -1, :] - fut_xy[None, :, -1, :], dim=-1
            )
            min_fde    = fde_per_sample.min(dim=0).values.mean()
            total_fde += min_fde.item()
            n_batches += 1

    return total_fde / n_batches


def validate_losses(student, teacher, val_loader, device, alpha, seed, max_batches=20):
    """Distillation losses on validation batches with fixed flow times / noise."""
    student.eval()
    torch.manual_seed(seed)
    tot = tot_gt = tot_kd = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break

            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            t                 = sample_flow_time(obs.shape[0], device=str(device))
            x_t, noise, v_tgt = forward_cfm(fut, t)
            v_pred            = student(obs, x_t, t, t_rel)
            v_teacher         = teacher(obs, x_t, t, t_rel)

            loss_gt = weighted_mse(v_pred, v_tgt).item()
            loss_kd = weighted_mse(v_pred, v_teacher).item()
            tot    += alpha * loss_gt + (1 - alpha) * loss_kd
            tot_gt += loss_gt
            tot_kd += loss_kd
            n_batches += 1

    return tot / n_batches, tot_gt / n_batches, tot_kd / n_batches


def validate_kl(student, teacher, val_loader, device, feat_std, seed,
                n_samples=20, max_batches=5):
    """
    KL(teacher ‖ student) between the distributions of sampled future positions.

    For each val window, draw n_samples futures from each model (same starting
    noise for both, so the comparison is paired), fit a 2-D Gaussian to the
    (x, y) positions in metres at every future step, and compute the closed-form
    Gaussian KL. Returns (KL at the final step, KL averaged over all steps),
    each averaged over windows.
    """
    student.eval()
    feat_std_xy = torch.tensor(feat_std[:2], device=device, dtype=torch.float32)
    kl_end_sum  = kl_path_sum = 0.0
    n_windows   = 0

    with torch.no_grad():
        for b, batch in enumerate(val_loader):
            if b >= max_batches:
                break

            obs   = batch["obs"].to(device)
            t_rel = batch["t_rel"].to(device)

            torch.manual_seed(seed + b)
            s_xy = euler_sample(student, obs, t_rel, n_samples=n_samples,
                                n_steps=20, device=str(device))[..., :2] * feat_std_xy
            torch.manual_seed(seed + b)
            t_xy = euler_sample(teacher, obs, t_rel, n_samples=n_samples,
                                n_steps=20, device=str(device))[..., :2] * feat_std_xy
            # (K, B, 43, 2) in metres; mean offset cancels in the KL, scale matters

            mu_t, cov_t = fit_gaussian(t_xy.double())
            mu_s, cov_s = fit_gaussian(s_xy.double())
            kl = gaussian_kl(mu_t, cov_t, mu_s, cov_s)    # (B, 43)

            kl_end_sum  += kl[:, -1].sum().item()
            kl_path_sum += kl.mean(dim=1).sum().item()
            n_windows   += kl.shape[0]

    return kl_end_sum / n_windows, kl_path_sum / n_windows


def load_teacher(teacher_cfg, teacher_ckpt, device):
    teacher = TrajectoryDiT(**teacher_cfg["model"]).to(device)
    ckpt    = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    state   = ckpt.get("ema_state") or ckpt.get("model_state")
    teacher.load_state_dict(state)
    teacher.eval()                      # disables dropout
    teacher.requires_grad_(False)       # frozen
    return teacher, ckpt.get("epoch"), ckpt.get("val_fde")


def append_metrics(path, row):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def train(
    nc_path,
    teacher_config,
    student_config,
    teacher_ckpt   = None,
    output_dir     = str(CHECKPOINT_DIR / "swi_dit_distill"),
    epochs         = 100,
    batch_size     = 64,
    lr             = 1e-4,
    weight_decay   = 0.01,
    grad_clip      = 1.0,
    warmup_steps   = 1000,
    alpha          = 0.5,
    seed           = 42,
    kl_samples     = 20,
    kl_batches     = 5,
    device         = "cuda",
    subset         = None,
):
    set_seed(seed)

    teacher_cfg  = load_config(teacher_config)
    student_cfg  = load_config(student_config)
    if not (teacher_ckpt or teacher_cfg.get("checkpoint")):
        raise ValueError(f"{teacher_config}: no 'checkpoint' in the teacher config and no --teacher_ckpt given")
    teacher_ckpt = resolve_path(teacher_ckpt or teacher_cfg["checkpoint"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in (("teacher_config.json", teacher_cfg), ("student_config.json", student_cfg)):
        with open(output_dir / name, "w") as f:
            json.dump(cfg, f, indent=2)
    metrics_path = output_dir / "metrics.csv"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} — CFM + SWI_DiT distillation "
          f"({teacher_cfg.get('name', 'teacher')} → {student_cfg.get('name', 'student')}, "
          f"alpha={alpha}, seed={seed})")

    # seed controls the aircraft-level train/val/test split
    train_loader, val_loader, test_loader = get_dataloaders(
        nc_path, batch_size=batch_size, seed=seed, subset=subset,
    )

    import netCDF4 as nc
    ds       = nc.Dataset(nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()

    # ── Teacher (frozen) ──────────────────────────────────────────────────
    teacher, teacher_epoch, teacher_ckpt_fde = load_teacher(teacher_cfg, teacher_ckpt, device)
    print(f"Teacher: {teacher_ckpt} (epoch={teacher_epoch}, ckpt val_fde={teacher_ckpt_fde}) "
          f"| {teacher_cfg['model']} | {sum(p.numel() for p in teacher.parameters()):,} params")

    # ── Student ───────────────────────────────────────────────────────────
    model = TrajectoryDiT(**student_cfg["model"]).to(device)
    ema   = EMA(model, decay=0.9999)
    print(f"Student: {student_cfg['model']} | {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_config = {
        "teacher_ckpt": str(teacher_ckpt),
        "alpha":        alpha,
        "seed":         seed,
        "subset":       subset,
        "batch_size":   batch_size,
        "lr":           lr,
        "epochs":       epochs,
    }

    start_epoch = 0
    best_fde    = float("inf")
    global_step = 0
    last_ckpt   = output_dir / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_fde    = ckpt.get("best_fde", best_fde)
        global_step = ckpt.get("global_step", global_step)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")

    # Teacher reference on the same val batches the student is scored on
    teacher_fde = validate(teacher, val_loader, device, feat_std, seed, max_batches=20)
    print(f"Teacher val minFDE (this split): {teacher_fde:.1f}m", flush=True)

    for epoch in range(start_epoch, epochs):
        torch.manual_seed(seed + epoch)    # same batch order / noise per epoch, even after resume
        epoch_start = time.time()
        epoch_lr    = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss = total_gt = total_kd = 0.0
        n_batches  = 0

        for batch in train_loader:
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            t                 = sample_flow_time(obs.shape[0], device=str(device))
            x_t, noise, v_tgt = forward_cfm(fut, t)
            v_pred            = model(obs, x_t, t, t_rel)

            with torch.no_grad():
                v_teacher = teacher(obs, x_t, t, t_rel)

            loss_gt = weighted_mse(v_pred, v_tgt)
            loss_kd = weighted_mse(v_pred, v_teacher)
            loss    = alpha * loss_gt + (1 - alpha) * loss_kd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * lr_scale

            ema.update(model)
            total_loss  += loss.item()
            total_gt    += loss_gt.item()
            total_kd    += loss_kd.item()
            n_batches   += 1
            global_step += 1

        scheduler.step()

        # ── Per-epoch student metrics (EMA weights) ───────────────────────
        val_loss, val_gt, val_kd = validate_losses(
            ema.shadow, teacher, val_loader, device, alpha, seed, max_batches=20
        )
        kl_end, kl_path = validate_kl(
            ema.shadow, teacher, val_loader, device, feat_std, seed,
            n_samples=kl_samples, max_batches=kl_batches,
        )
        rough_fde = validate(ema.shadow, val_loader, device, feat_std, seed, max_batches=5)

        metrics = {
            "epoch":            epoch + 1,
            "lr":               f"{epoch_lr:.3e}",
            "train_loss":       f"{total_loss / n_batches:.6f}",
            "train_loss_gt":    f"{total_gt / n_batches:.6f}",
            "train_loss_kd":    f"{total_kd / n_batches:.6f}",
            "val_loss":         f"{val_loss:.6f}",
            "val_loss_gt":      f"{val_gt:.6f}",
            "val_loss_kd":      f"{val_kd:.6f}",
            "val_kl_endpoint":  f"{kl_end:.4f}",
            "val_kl_path":      f"{kl_path:.4f}",
            "val_minfde_rough": f"{rough_fde:.1f}",
        }

        checkpoint = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "ema_state":       ema.shadow.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step":     global_step,
            "best_fde":        best_fde,
            "teacher_val_fde": teacher_fde,
            "teacher_config":  teacher_cfg,
            "student_config":  student_cfg,
            "train_config":    train_config,
            "metrics":         metrics,
            "val_fde":         None,
        }

        # ── Every 5 epochs: proper FDE (20 batches ~1280 samples) ─────────
        fde_str = f"rough FDE {rough_fde:.1f}m"
        if (epoch + 1) % 5 == 0:
            proper_fde = validate(ema.shadow, val_loader, device, feat_std, seed, max_batches=20)
            metrics["val_minfde"] = f"{proper_fde:.1f}"
            fde_str += f" | proper FDE {proper_fde:.1f}m (teacher {teacher_fde:.1f}m)"

            checkpoint["val_fde"] = proper_fde
            if proper_fde < best_fde:
                best_fde = proper_fde
                checkpoint["best_fde"] = best_fde
                torch.save(checkpoint, output_dir / "best.pt")
                fde_str += "  ✓ best"

        metrics["epoch_time_s"] = f"{time.time() - epoch_start:.1f}"
        append_metrics(metrics_path, metrics)
        torch.save(checkpoint, output_dir / "last.pt")

        print(f"Epoch {epoch+1:03d} | "
              f"train {metrics['train_loss']} (gt {metrics['train_loss_gt']}, kd {metrics['train_loss_kd']}) | "
              f"val {metrics['val_loss']} (gt {metrics['val_loss_gt']}, kd {metrics['val_loss_kd']}) | "
              f"KL end {metrics['val_kl_endpoint']} path {metrics['val_kl_path']} | {fde_str}",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=str, required=True,
                         help="Path to the .nc trajectory dataset")
    parser.add_argument("--teacher_config", type=str, required=True,
                         help="JSON config for the teacher (model kwargs + checkpoint), "
                              "e.g. configs/swi_dit_teacher.json")
    parser.add_argument("--student_config", type=str, required=True,
                         help="JSON config for the student (model kwargs), "
                              "e.g. configs/swi_dit_student.json")
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                         help="Override the checkpoint path given in the teacher config")
    parser.add_argument("--output_dir", type=str, default=str(CHECKPOINT_DIR / "swi_dit_distill"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Weight on the ground-truth CFM loss; (1 - alpha) goes to the teacher-matching loss")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seeds the train/val/test split and all RNGs")
    parser.add_argument("--kl_samples", type=int, default=20,
                         help="Samples per window per model for the KL metric")
    parser.add_argument("--kl_batches", type=int, default=5,
                         help="Validation batches used for the KL metric")
    parser.add_argument("--subset", type=int, default=None,
                         help="Limit to N samples; omit for the full dataset")
    args = parser.parse_args()

    train(
        nc_path        = args.nc_path,
        teacher_config = args.teacher_config,
        student_config = args.student_config,
        teacher_ckpt   = args.teacher_ckpt,
        output_dir     = args.output_dir,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        alpha          = args.alpha,
        seed           = args.seed,
        kl_samples     = args.kl_samples,
        kl_batches     = args.kl_batches,
        subset         = args.subset,
    )
