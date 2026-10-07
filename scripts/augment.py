"""Generate augmented training windows from the raw recordings.

Reads every raw recording in data/raw/, cuts it into 3 s gesture windows
(same segmentation train_demo.py uses), and writes N randomly perturbed
copies of each window to:

    data/augmented/<gesture>/<source>_w<window>_a<copy>.csv
    columns: t, ax, ay, az (same schema as data/raw)
    data/augmented/manifest.csv  (one row per file, with the parameters used)

Each copy applies all of these IMU-appropriate transforms with random
strength:
  * rotation    -- small 3-D rotation (up to +/-15 deg per axis): the phone
                   held at a slightly different angle in the hand
  * scaling     -- whole-window magnitude x0.8-1.2: a weaker/stronger motion
  * time warp   -- stretch/squeeze in time x0.85-1.15: a slower/faster gesture
  * time shift  -- move the gesture up to +/-0.25 s within the window:
                   imperfect segmentation
  * jitter      -- Gaussian noise, sigma 0.05-0.20 m/s^2: sensor noise

Usage:
    python scripts/augment.py                 # 5 copies per window, seed 0
    python scripts/augment.py --copies 10 --seed 1
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from common import AUG_DIR, FS, RAW_DIR, WINDOW, load_raw_windows, window_to_frame


def rotation_matrix(rng: np.random.Generator, max_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ax, ay, az = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx, np.rad2deg([ax, ay, az])


def time_warp(w: np.ndarray, factor: float) -> np.ndarray:
    """Resample so the motion takes `factor` times as long, keeping it centered."""
    n = len(w)
    src = (np.arange(n) - n / 2) / factor + n / 2  # where each output sample reads from
    src = np.clip(src, 0, n - 1)
    return np.stack([np.interp(src, np.arange(n), w[:, i]) for i in range(3)], axis=1)


def time_shift(w: np.ndarray, shift: int) -> np.ndarray:
    idx = np.clip(np.arange(len(w)) - shift, 0, len(w) - 1)  # edge-pad instead of wrapping
    return w[idx]


def augment_window(w: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    rot, angles = rotation_matrix(rng, 15)
    scale = rng.uniform(0.8, 1.2)
    warp = rng.uniform(0.85, 1.15)
    shift = int(rng.integers(-int(0.25 * FS), int(0.25 * FS) + 1))
    sigma = rng.uniform(0.05, 0.20)

    gravity = w.mean(axis=0)                      # scale only the motion, not gravity
    out = gravity + (w - gravity) * scale
    out = time_shift(time_warp(out, warp), shift)
    out = out @ rot.T
    out = out + rng.normal(0, sigma, out.shape)
    params = {
        "rot_x_deg": round(angles[0], 2), "rot_y_deg": round(angles[1], 2), "rot_z_deg": round(angles[2], 2),
        "scale": round(scale, 3), "time_warp": round(warp, 3), "shift_samples": shift,
        "noise_sigma": round(sigma, 3),
    }
    return out, params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copies", type=int, default=5, help="augmented copies per raw window (default 5)")
    ap.add_argument("--seed", type=int, default=0, help="random seed, for reproducible output (default 0)")
    args = ap.parse_args()

    windows, meta = load_raw_windows(RAW_DIR)
    assert windows.shape[1] == WINDOW
    rng = np.random.default_rng(args.seed)

    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)  # always regenerate from scratch so stale files can't linger
    rows = []
    for w, m in zip(windows, meta):
        for c in range(args.copies):
            aug, params = augment_window(w, rng)
            name = f"{m['source']}_w{m['window']:02d}_a{c:02d}.csv"
            path = AUG_DIR / m["gesture"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            window_to_frame(aug).to_csv(path, index=False)
            rows.append({"file": f"data/augmented/{m['gesture']}/{name}", "gesture": m["gesture"],
                         "person": m["person"], "session": m["session"], "source": m["source"],
                         "window": m["window"], "copy": c, **params})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(AUG_DIR / "manifest.csv", index=False)
    print(f"{len(windows)} raw windows -> {len(manifest)} augmented windows in {AUG_DIR}")
    print(manifest["gesture"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
