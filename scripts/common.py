"""Shared helpers: file naming, loading raw recordings, resampling and
splitting each recording into fixed-length gesture windows.

Every other script in scripts/ imports from here so that the label scheme,
sample rate and window length are defined in exactly one place.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
AUG_DIR = REPO_ROOT / "data" / "augmented"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"

CLASSES = ["wing", "ring", "slope", "idle"]
GESTURE_CLASSES = [c for c in CLASSES if c != "idle"]

FS = 100            # Hz: every recording is resampled to this rate on load
WINDOW_S = 3.0      # seconds per training example
WINDOW = int(FS * WINDOW_S)
REPS_PER_RECORDING = 10   # every gesture recording was captured as 10 repetitions
MERGE_GAPS_S = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
IDLE_STRIDE = WINDOW  # idle recordings are cut into back-to-back, non-overlapping windows

COLUMNS = ["t", "ax", "ay", "az"]
G = 9.80665

# <gesture>_p<person>_s<session>_<take>, e.g. ring_p2_s1_03
NAME_RE = re.compile(r"^(?P<gesture>[a-z]+)_p(?P<person>\d+)_s(?P<session>\d+)_(?P<take>\d+)$")


def parse_name(stem: str) -> dict:
    """Split a recording's file stem into its label and metadata."""
    m = NAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"'{stem}' does not follow <gesture>_p<person>_s<session>_<take> (e.g. ring_p2_s1_03)"
        )
    info = m.groupdict()
    if info["gesture"] not in CLASSES:
        raise ValueError(f"'{stem}': unknown gesture '{info['gesture']}', expected one of {CLASSES}")
    return {
        "gesture": info["gesture"],
        "person": f"p{int(info['person'])}",
        "session": int(info["session"]),
        "take": int(info["take"]),
    }


def raw_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(raw_dir.glob("*/*.csv"))


def load_recording(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[COLUMNS].sort_values("t").reset_index(drop=True)


def resample(df: pd.DataFrame, fs: int = FS) -> np.ndarray:
    """Linearly interpolate onto a uniform fs grid -> array of shape (n, 3).

    Phone sensors don't sample perfectly evenly, so this is done on load
    rather than baked into the raw files.
    """
    t = df["t"].to_numpy(dtype=float)
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return np.stack([np.interp(grid, t, df[c].to_numpy(dtype=float)) for c in ("ax", "ay", "az")], axis=1)


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centered running mean, edge-padded so the ends aren't pulled toward zero."""
    n = max(1, n)
    kernel = np.ones(n) / n

    def smooth(v: np.ndarray) -> np.ndarray:
        padded = np.pad(v, (n // 2, n - 1 - n // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    if x.ndim == 1:
        return smooth(x)
    return np.stack([smooth(x[:, i]) for i in range(x.shape[1])], axis=1)


def motion_energy(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """Magnitude of the signal with its slow (gravity/orientation) part removed."""
    dynamic = sig - _moving_average(sig, 2 * fs)      # subtract 2 s running mean per axis
    return _moving_average(np.linalg.norm(dynamic, axis=1), int(0.1 * fs))


def _bursts(active: np.ndarray) -> list[list[int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    return [list(r) for r in edges.reshape(-1, 2)]


def find_gesture_regions(sig: np.ndarray, fs: int = FS, expected: int = REPS_PER_RECORDING) -> list[tuple[int, int]]:
    """Return (start, end) sample indices of each repetition in a recording.

    A burst of motion above the threshold is one repetition. Some gestures
    (slope's corner, wing's reversals) briefly slow down mid-gesture, so
    bursts closer together than a merge gap are joined. Pauses between
    repetitions varied between recordings, so the merge gap is chosen per
    recording: the smallest gap in MERGE_GAPS_S whose repetition count is
    closest to `expected`.
    """
    energy = motion_energy(sig, fs)
    threshold = max(1.0, 0.25 * np.percentile(energy, 99))  # m/s^2
    bursts = _bursts(energy > threshold)

    best = None
    for gap_s in MERGE_GAPS_S:
        merged: list[list[int]] = []
        for b in bursts:
            if merged and b[0] - merged[-1][1] < int(gap_s * fs):
                merged[-1][1] = b[1]
            else:
                merged.append(list(b))
        regions = [(s, e) for s, e in merged if e - s >= int(0.3 * fs)]
        if best is None or abs(len(regions) - expected) < abs(len(best) - expected):
            best = regions
    return best


def _centered_window(sig: np.ndarray, center: int, length: int = WINDOW) -> np.ndarray:
    start = center - length // 2
    idx = np.clip(np.arange(start, start + length), 0, len(sig) - 1)  # edge-pad near the ends
    return sig[idx]


def recording_windows(sig: np.ndarray, gesture: str) -> list[np.ndarray]:
    """Cut one resampled recording into WINDOW-length examples."""
    if gesture == "idle":
        return [sig[s:s + WINDOW] for s in range(0, len(sig) - WINDOW + 1, IDLE_STRIDE)]
    return [_centered_window(sig, (s + e) // 2) for s, e in find_gesture_regions(sig)]


def load_raw_windows(raw_dir: Path = RAW_DIR) -> tuple[np.ndarray, list[dict]]:
    """Load every raw recording and return (windows, metadata per window).

    windows has shape (n, WINDOW, 3). Each metadata dict has gesture,
    person, session, take and source file.
    """
    windows, meta = [], []
    for path in raw_files(raw_dir):
        info = parse_name(path.stem)
        if path.parent.name != info["gesture"]:
            raise ValueError(f"{path}: folder '{path.parent.name}' disagrees with file name")
        for k, w in enumerate(recording_windows(resample(load_recording(path)), info["gesture"])):
            windows.append(w)
            meta.append({**info, "source": path.stem, "window": k})
    if not windows:
        raise SystemExit(f"No recordings found under {raw_dir}")
    return np.stack(windows), meta


def window_to_frame(w: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "t": np.round(np.arange(len(w)) / FS, 4),
        "ax": np.round(w[:, 0], 5),
        "ay": np.round(w[:, 1], 5),
        "az": np.round(w[:, 2], 5),
    })


def write_manifest(raw_dir: Path = RAW_DIR, out: Path = MANIFEST_PATH) -> pd.DataFrame:
    """One row per raw recording, including how many windows it yields."""
    rows = []
    for path in raw_files(raw_dir):
        info = parse_name(path.stem)
        df = load_recording(path)
        duration = float(df["t"].iloc[-1] - df["t"].iloc[0])
        n_windows = len(recording_windows(resample(df), info["gesture"]))
        rows.append({
            "file": path.relative_to(REPO_ROOT).as_posix(),
            **info,
            "n_samples": len(df),
            "duration_s": round(duration, 2),
            "mean_rate_hz": round((len(df) - 1) / duration, 1) if duration > 0 else 0,
            "n_windows": n_windows,
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out, index=False)
    return manifest
