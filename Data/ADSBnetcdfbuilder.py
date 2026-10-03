import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import tqdm  # ADDED: progress bars like mentor's version

def load_file(filepath):
    df = pd.read_csv(filepath)
    df_filtered = df[['time_position', 'icao24', 'callsign', 'longitude', 'latitude', 
                       'geo_altitude', 'velocity', 'true_track', 'vertical_rate', 'on_ground']].copy()

    df_filtered["time_position"] = pd.to_numeric(df_filtered["time_position"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["time_position", "velocity"])
    df_filtered["geo_altitude"]  = df_filtered["geo_altitude"].fillna(0)
    df_filtered["vertical_rate"] = df_filtered["vertical_rate"].fillna(0)

    df_filtered["callsign"] = df_filtered.groupby("icao24")["callsign"].ffill().bfill()
    df_filtered = df_filtered.dropna(subset=['callsign'])

    df_cleaned = df_filtered.drop_duplicates(subset=['icao24', 'time_position'])
    return df_cleaned

def convertToCartesian(df):
    radius = 6371000.0
    originLong = np.radians(-122.3790)   # CHANGED: matched mentor's exact SFO reference
    originLat  = np.radians(37.6213)     # CHANGED: matched mentor's exact SFO reference
    longitudeVal = np.radians(df['longitude'].to_numpy())
    latitudeVal  = np.radians(df['latitude'].to_numpy())
    altitude  = df['geo_altitude'].to_numpy()
    track_rad = np.radians(df["true_track"].to_numpy())

    df_copy = df.copy()

    # Velocities
    df_copy["vx"] = df["velocity"] * np.sin(track_rad)
    df_copy["vy"] = df["velocity"] * np.cos(track_rad)
    df_copy["vz"] = df["vertical_rate"]

    # Positions
    df_copy["x"] = (radius + altitude) * np.cos(latitudeVal) * (longitudeVal - originLong)
    df_copy["y"] = radius * (latitudeVal - originLat)
    df_copy["z"] = altitude

    df_copy = df_copy.drop(columns=["latitude", "longitude", "geo_altitude",
                                     "velocity", "true_track", "vertical_rate"])  # ADDED: drop raw columns after decomposition like mentor does
    return df_copy

def segment_aircraft(group, seq_len):
    n = len(group)
    segments = []
    start = 0
    for i in range(1, n):
        dt = group["time_position"].iat[i] - group["time_position"].iat[i - 1]
        same_callsign = group["callsign"].iat[i] == group["callsign"].iat[i - 1]
        if dt > 120 or not same_callsign:
            seg = group.iloc[start:i]
            if len(seg) >= seq_len and not seg["on_ground"].all():
                segments.append((start, i))
            start = i
    seg = group.iloc[start:n]
    if len(seg) >= seq_len and not seg["on_ground"].all():
        segments.append((start, n))
    return segments

def extract_windows(df, seq_len, offset):
    df = df.sort_values(["icao24", "time_position"]).reset_index(drop=True)
    sequences, icao_list, callsign_list, timestamps_list = [], [], [], []

    for icao, group in tqdm.tqdm(df.groupby("icao24", sort=False),   # ADDED: tqdm
                                  desc="Aircraft", leave=False):
        group = group.reset_index(drop=True)
        if len(group) < seq_len:
            continue

        for seg_start, seg_end in segment_aircraft(group, seq_len):
            seg = group.iloc[seg_start:seg_end].reset_index(drop=True)
            for ws in range(0, len(seg) - seq_len + 1, offset):
                window = seg.iloc[ws: ws + seq_len]
                if window["on_ground"].all():
                    continue
                features = window[["x","y","z","vx","vy","vz"]].values.astype(np.float32)
                if (features[:, 2] < -50).any():
                    continue
                sequences.append(features)
                icao_list.append(icao)
                callsign_list.append(window["callsign"].iat[0])
                timestamps_list.append(window["time_position"].values)

    return sequences, icao_list, callsign_list, timestamps_list

def to_delta_xy(data):
    N, seq_len, _ = data.shape
    delta_xy = np.zeros((N, seq_len, 2), dtype=np.float32)
    delta_xy[:, 1:, :] = data[:, 1:, :2] - data[:, :-1, :2]

    traj = np.concatenate([delta_xy, data[:, :, 2:]], axis=-1)
    origin = data[:, 0, :2].astype(np.float32)
    return traj, origin

def compute_stats(data, traj, times):
    # CHANGED: now takes raw `data` (absolute x,y,z,vx,vy,vz) AND `traj` (delta xy)
    # to compute the full set of stats your mentor saves
    stats = {}

    # --- stats your mentor saves ---

    # Global per-feature mean/std over raw absolute values (mentor's feature_mean/std)
    flat64 = data.reshape(-1, 6).astype(np.float64)
    stats["feature_mean"] = flat64.mean(axis=0)
    stats["feature_std"]  = flat64.std(axis=0)

    # Step-to-step deltas on raw absolute data (mentor's delta_mean/std)
    diffs64 = (data[:, 1:, :] - data[:, :-1, :]).reshape(-1, 6).astype(np.float64)
    stats["delta_mean"] = diffs64.mean(axis=0)
    stats["delta_std"]  = diffs64.std(axis=0)

    # Relative time from start of each window (mentor's t_rel_mean/std)
    t_rel64 = (times - times[:, 0:1]).astype(np.float64)
    stats["t_rel_mean"] = t_rel64.mean()
    stats["t_rel_std"]  = t_rel64.std()

    # Time gap between consecutive timesteps (mentor's dt_mean/std)
    dt_vals = (times[:, 1:] - times[:, :-1]).ravel().astype(np.float64)
    dt_vals = dt_vals[(dt_vals > 0) & (dt_vals < 120)]
    stats["dt_mean"] = dt_vals.mean()
    stats["dt_std"]  = dt_vals.std()

    # --- stats specific to your delta-xy representation ---

    # Delta xy stats (excluding t=0 which is always 0 by convention)
    xy_diffs = traj[:, 1:, :2].reshape(-1, 2).astype(np.float64)
    stats["pos_delta_mean"] = xy_diffs.mean(axis=0)
    stats["pos_delta_std"]  = xy_diffs.std(axis=0)

    # Absolute z stats
    z_flat = traj[:, :, 2].ravel().astype(np.float64)
    stats["z_mean"] = z_flat.mean()
    stats["z_std"]  = z_flat.std()

    # Absolute velocity stats
    vel_flat = traj[:, :, 3:].reshape(-1, 3).astype(np.float64)
    stats["vel_mean"] = vel_flat.mean(axis=0)
    stats["vel_std"]  = vel_flat.std(axis=0)

    return stats

def write_netcdf(output_path, traj, origin, icao_list, callsign_list, times, stats, seq_len, offset):
    N = traj.shape[0]
    ds = nc.Dataset(output_path, "w", format="NETCDF4")
    try:
        ds.createDimension("sample",   N)
        ds.createDimension("sequence", seq_len)
        ds.createDimension("feature",  6)
        ds.createDimension("xy",       2)

        ds.createVariable("icao24",    str, ("sample",))[:] = np.array(icao_list,    dtype="S")
        ds.createVariable("callsign",  str, ("sample",))[:] = np.array(callsign_list, dtype="S")
        ds.createVariable("timestamps", "f8", ("sample", "sequence"))[:] = times

        tv = ds.createVariable("trajectory", "f4", ("sample", "sequence", "feature"),
                                zlib=True, complevel=4)
        tv[:] = traj
        tv.feature_names = "x,y,z,vx,vy,vz"

        ds.createVariable("origin", "f4", ("sample", "xy"))[:] = origin

        for key, val in stats.items():
            setattr(ds, key, val.tolist() if hasattr(val, "tolist") else float(val))

        ds.seq_len = seq_len
        ds.offset  = offset
        ds.description = f"ADS-B trajectory sequences ({seq_len} timesteps, Cartesian, absolute) — Bay Area"
    finally:
        ds.close()

# CHANGED: build() now processes files one at a time with buffer_tail stitching
# instead of concat-all-then-window, matching mentor's approach
def build(file_paths, seq_len, offset, output_path):
    all_seq, all_icao, all_cs, all_ts = [], [], [], []
    buffer_tail = None   # tail rows carried over from previous file

    for file_path in tqdm.tqdm(file_paths, desc="Files"):
        print(f"\nLoading {file_path.name}...")
        df = convertToCartesian(load_file(file_path))

        # ADDED: stitch tail of previous file onto front of this one
        # so flights crossing a file boundary still get valid windows
        if buffer_tail is not None:
            df = pd.concat([buffer_tail, df], ignore_index=True)

        # ADDED: save last (seq_len - 1) timestamps as tail for next iteration
        tail_ts     = df["time_position"].unique()[-(seq_len - 1):].tolist()
        buffer_tail = df[df["time_position"].isin(tail_ts)].copy()

        print(f"Extracting sequences...")
        seq, icao, cs, ts = extract_windows(df, seq_len, offset)
        print(f"  -> {len(seq):,} sequences")

        all_seq.extend(seq)
        all_icao.extend(icao)
        all_cs.extend(cs)
        all_ts.extend(ts)

    if not all_seq:
        print("No sequences generated.")
        return

    data  = np.array(all_seq)   # (N, seq_len, 6) absolute x,y,z,vx,vy,vz
    all_seq.clear()             # free per-window arrays before building the rest
    times = np.array(all_ts)    # (N, seq_len)
    all_ts.clear()

    traj, origin = to_delta_xy(data)               # traj is now dx,dy,z,vx,vy,vz
    stats = compute_stats(data, traj, times)        # pass both raw and delta
    del traj

    # CHANGED: store absolute x,y,z,vx,vy,vz — ADSBdataset normalizes `trajectory`
    # with feature_mean/std (computed on absolute data) and the UI plots it directly
    write_netcdf(output_path, data, origin, all_icao, all_cs, times, stats, seq_len, offset)
    print(f"\nSaved {data.shape[0]:,} samples -> {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input",    nargs="+", help="Input CSV file(s)")
    parser.add_argument("output",   help="Output .nc file")
    parser.add_argument("--seq-len", type=int, default=86)
    parser.add_argument("--offset",  type=int, default=1)
    args = parser.parse_args()

    build(
        file_paths=[Path(p) for p in args.input],
        seq_len=args.seq_len,
        offset=args.offset,
        output_path=Path(args.output),
    )