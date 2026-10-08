"""Convert Sensor Logger exports into this dataset's raw CSV format.

Each Sensor Logger recording exports as a .zip (or an unzipped folder)
containing one CSV per sensor. This script reads Accelerometer.csv
(gravity removed) plus Gravity.csv, adds them back together to get the
total acceleration a phone actually feels, and writes one tidy file:

    data/raw/<gesture>/<gesture>_p<person>_s<session>_<take>.csv
    columns: t (s since start), ax, ay, az (m/s^2, gravity included)

Name each export <gesture>_p<person>_s<session>_<take> before importing
(e.g. ring_p2_s1_03.zip), or pass --gesture/--person/--session for a
single unrenamed export.

Usage:
    python scripts/import_sensorlogger.py path/to/exports/*.zip
    python scripts/import_sensorlogger.py 2026-10-07_14-03-22.zip --gesture ring --person 2 --session 1
    python scripts/import_sensorlogger.py exports/idle_p1_s2_01.zip --max-seconds 40
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASSES, G, RAW_DIR, parse_name, write_manifest


def _read_sensor_csvs(src: Path) -> dict[str, pd.DataFrame]:
    """Map sensor file name (e.g. 'Accelerometer') -> DataFrame."""
    tables = {}
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    tables[Path(name).stem] = pd.read_csv(io.BytesIO(zf.read(name)))
    elif src.is_dir():
        for f in src.rglob("*.csv"):
            tables[f.stem] = pd.read_csv(f)
    else:
        raise SystemExit(f"{src}: expected a Sensor Logger .zip or an unzipped export folder")
    return tables


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    if "seconds_elapsed" in df.columns:
        return df["seconds_elapsed"].to_numpy(dtype=float)
    return (df["time"].to_numpy(dtype=float) - df["time"].iloc[0]) * 1e-9  # ns -> s


def _xyz(df: pd.DataFrame) -> np.ndarray:
    return df[["x", "y", "z"]].to_numpy(dtype=float)


def total_acceleration(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "Accelerometer" in tables and "Gravity" in tables:
        acc, grav = tables["Accelerometer"], tables["Gravity"]
        t = _time_seconds(acc)
        gt = _time_seconds(grav)
        g_on_acc = np.stack([np.interp(t, gt, grav[c]) for c in ("x", "y", "z")], axis=1)
        xyz = _xyz(acc) + g_on_acc
    elif "TotalAcceleration" in tables:
        tot = tables["TotalAcceleration"]
        t, xyz = _time_seconds(tot), _xyz(tot)
    else:
        raise SystemExit(f"need Accelerometer.csv + Gravity.csv (found: {sorted(tables)})")

    if np.median(np.linalg.norm(xyz, axis=1)) < 3:  # some exports are in g, not m/s^2
        xyz = xyz * G
    t = t - t[0]
    return pd.DataFrame({"t": np.round(t, 5), "ax": np.round(xyz[:, 0], 5),
                         "ay": np.round(xyz[:, 1], 5), "az": np.round(xyz[:, 2], 5)})


def _next_take(gesture: str, person: int, session: int) -> int:
    existing = list((RAW_DIR / gesture).glob(f"{gesture}_p{person}_s{session}_*.csv"))
    return max([parse_name(p.stem)["take"] for p in existing], default=0) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path, help="Sensor Logger .zip files or unzipped folders")
    ap.add_argument("--gesture", choices=CLASSES, help="label for a single unrenamed export")
    ap.add_argument("--person", type=int, help="person number (1, 2, ...) for a single unrenamed export")
    ap.add_argument("--session", type=int, help="session number for a single unrenamed export")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing raw file of the same name")
    ap.add_argument("--max-seconds", type=float,
                    help="keep only the first N seconds (used to trim over-long idle recordings)")
    args = ap.parse_args()

    manual = args.gesture is not None
    if manual and (args.person is None or args.session is None or len(args.sources) != 1):
        ap.error("--gesture needs --person and --session, and exactly one source")

    for src in args.sources:
        if manual:
            take = _next_take(args.gesture, args.person, args.session)
            stem = f"{args.gesture}_p{args.person}_s{args.session}_{take:02d}"
        else:
            stem = src.stem if src.suffix.lower() == ".zip" else src.name
            parse_name(stem)  # raises with a clear message if misnamed
        gesture = parse_name(stem)["gesture"]
        out = RAW_DIR / gesture / f"{stem}.csv"
        if out.exists() and not args.overwrite:
            print(f"skip   {out.relative_to(RAW_DIR.parent.parent)} (exists; use --overwrite)")
            continue
        df = total_acceleration(_read_sensor_csvs(src))
        if args.max_seconds:
            df = df[df["t"] <= args.max_seconds]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote  {out.relative_to(RAW_DIR.parent.parent)}  ({len(df)} rows, {df['t'].iloc[-1]:.1f} s)")

    manifest = write_manifest()
    print(f"\nmanifest: {len(manifest)} recordings")
    print(manifest.groupby("gesture")[["n_windows"]].sum().rename(columns={"n_windows": "windows"}).to_string())


if __name__ == "__main__":
    main()
