import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_netcdf(filepath):
    ds = nc.Dataset(filepath, "r")
    trajectory = ds["trajectory"][:]
    timestamps = ds["timestamps"][:]
    icao24 = list(ds["icao24"][:])
    stats = {
        "feature_mean": np.array(ds.feature_mean),
        "feature_std":  np.array(ds.feature_std),
        "t_rel_mean":   float(ds.t_rel_mean),
        "t_rel_std":    float(ds.t_rel_std),
        "dt_mean":      float(ds.dt_mean),
        "dt_std":       float(ds.dt_std),
    }
    ds.close()
    return trajectory, timestamps, icao24, stats


def split_by_icao(icao24, train_frac=0.85, val_frac=0.10):
    # sorted: set order depends on per-process string hashing, which would make
    # the seeded shuffle (and therefore the split) differ between runs/resumes
    unique_icao = sorted(set(icao24))
    np.random.shuffle(unique_icao)
    n       = len(unique_icao)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train_icao = set(unique_icao[:n_train])
    val_icao   = set(unique_icao[n_train : n_train + n_val])
    test_icao  = set(unique_icao[n_train + n_val:])
    train_idx = [i for i, ic in enumerate(icao24) if ic in train_icao]
    val_idx   = [i for i, ic in enumerate(icao24) if ic in val_icao]
    test_idx  = [i for i, ic in enumerate(icao24) if ic in test_icao]
    return train_idx, val_idx, test_idx


def normalize(trajectory, stats):
    traj = trajectory.copy().astype(np.float32)
    traj = (traj - stats["feature_mean"]) / (stats["feature_std"] + 1e-8)
    return traj


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory, timestamps, indices, stats, obs_len=43):
        self.trajectory = trajectory
        self.timestamps = timestamps
        self.indices    = indices
        self.stats      = stats
        self.obs_len    = obs_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        traj     = self.trajectory[real_idx]
        traj     = normalize(traj, self.stats)
        obs      = traj[:self.obs_len]
        fut      = traj[self.obs_len:]
        ts       = self.timestamps[real_idx]
        t_rel    = ts - ts[0]
        t_rel    = (t_rel - self.stats["t_rel_mean"]) / (self.stats["t_rel_std"] + 1e-8)
        return {
            "obs":   torch.tensor(obs,   dtype=torch.float32),
            "fut":   torch.tensor(fut,   dtype=torch.float32),
            "t_rel": torch.tensor(t_rel, dtype=torch.float32),
        }


def get_dataloaders(filepath, batch_size=64, obs_len=43, seed=42, subset=None, num_workers=4):
    np.random.seed(seed)

    # BUG FIX: was calling load_netcdf twice — removed duplicate call
    trajectory, timestamps, icao24, stats = load_netcdf(filepath)

    if subset is not None:
        trajectory = trajectory[:subset]
        timestamps = timestamps[:subset]
        icao24     = icao24[:subset]

    train_idx, val_idx, test_idx = split_by_icao(icao24)

    train_ds = TrajectoryDataset(trajectory, timestamps, train_idx, stats, obs_len)
    val_ds   = TrajectoryDataset(trajectory, timestamps, val_idx,   stats, obs_len)
    test_ds  = TrajectoryDataset(trajectory, timestamps, test_idx,  stats, obs_len)

    # ADDED: num_workers=4 and pin_memory=True for faster data loading
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader