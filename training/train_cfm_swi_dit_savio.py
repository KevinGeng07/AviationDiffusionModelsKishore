import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataLoaders.ADSBdataset import get_dataloaders
from models.SWI_DiT import TrajectoryDiT
from models.cfm import sample_flow_time, forward_cfm, euler_sample

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


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


def validate(ema_model, val_loader, device, feat_std, max_batches=20):
    ema_model.shadow.eval()
    total_fde = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break

            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            preds = euler_sample(ema_model.shadow, obs, t_rel,
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


def train(
    nc_path,
    output_dir   = str(CHECKPOINT_DIR / "swi_dit"),
    epochs       = 100,
    batch_size   = 64,
    lr           = 1e-4,
    weight_decay = 0.01,
    grad_clip    = 1.0,
    warmup_steps = 1000,
    d_model      = 256,
    n_heads      = 8,
    n_layers     = 6,
    device       = "cuda",
    subset       = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} — CFM + SWI_DiT")

    train_loader, val_loader, test_loader = get_dataloaders(
        nc_path, batch_size=batch_size, subset=subset,
    )

    import netCDF4 as nc
    ds       = nc.Dataset(nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()

    model = TrajectoryDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    ema   = EMA(model, decay=0.9999)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    last_ckpt   = output_dir / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")

    best_fde    = float("inf")
    global_step = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            t                 = sample_flow_time(obs.shape[0], device=str(device))
            x_t, noise, v_tgt = forward_cfm(fut, t)
            v_pred            = model(obs, x_t, t, t_rel)

            fut_len    = fut.shape[1]
            loss_plain = nn.functional.mse_loss(v_pred, v_tgt, reduction="none")
            weights    = torch.ones(fut_len, device=device)
            weights[-fut_len // 4:] = 2.0
            loss = (loss_plain * weights[None, :, None]).mean()

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
            n_batches   += 1
            global_step += 1

        avg_loss = total_loss / n_batches
        scheduler.step()

        # ── Every epoch: quick rough FDE (5 batches ~320 samples) ─────────
        rough_fde = validate(ema, val_loader, device, feat_std, max_batches=5)

        # ── Every 5 epochs: proper FDE (20 batches ~1280 samples) ─────────
        if (epoch + 1) % 5 == 0:
            proper_fde = validate(ema, val_loader, device, feat_std, max_batches=20)
            print(f"Epoch {epoch+1:03d} | loss {avg_loss:.4f} | "
                  f"rough FDE {rough_fde:.1f}m | proper FDE {proper_fde:.1f}m")

            checkpoint = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.shadow.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_fde":         proper_fde,
            }
            torch.save(checkpoint, output_dir / "last.pt")

            if proper_fde < best_fde:
                best_fde = proper_fde
                torch.save(checkpoint, output_dir / "best.pt")
                print(f"  ✓ best model saved (minFDE {best_fde:.1f}m)")
        else:
            print(f"Epoch {epoch+1:03d} | loss {avg_loss:.4f} | rough FDE {rough_fde:.1f}m")

            checkpoint = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.shadow.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_fde":         None,
            }
            torch.save(checkpoint, output_dir / "last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=str, required=True,
                         help="Path to the .nc trajectory dataset")
    parser.add_argument("--output_dir", type=str, default=str(CHECKPOINT_DIR / "swi_dit"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--subset", type=int, default=None,
                         help="Limit to N samples; omit for the full dataset")
    args = parser.parse_args()

    train(
        nc_path    = args.nc_path,
        output_dir = args.output_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        d_model    = args.d_model,
        n_layers   = args.n_layers,
        subset     = args.subset,
    )
