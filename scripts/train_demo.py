"""Train and evaluate a gesture classifier on the dataset.

Pipeline:
  1. Load every raw recording and cut it into 3 s windows (scripts/common.py).
  2. Optionally add the augmented windows from data/augmented/ (run
     scripts/augment.py first).
  3. Turn each window into a feature vector: per-axis and magnitude
     statistics plus a coarse 10 Hz copy of the signal.
  4. Cross-validate a random forest with leave-one-person-out splits, so the
     model is always tested on a person it never saw. With only one person
     it falls back to leave-one-session-out, then to holding out whole
     recordings. Augmented windows are only ever used for training, and
     only those derived from the training fold's own recordings, so nothing
     from the test fold leaks in.
  5. Print per-fold accuracy, a combined classification report and confusion
     matrix, then fit on everything and save the model to
     models/gesture_rf.joblib.

Usage:
    python scripts/train_demo.py                    # raw + augmented (if present)
    python scripts/train_demo.py --no-augmented     # raw only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import AUG_DIR, CLASSES, FS, REPO_ROOT, WINDOW, load_raw_windows

MODEL_PATH = REPO_ROOT / "models" / "gesture_rf.joblib"


def features(w: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(w, axis=1, keepdims=True)
    x = np.hstack([w, mag])                       # (WINDOW, 4)
    dyn = x - x.mean(axis=0)
    stats = np.concatenate([
        x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0),
        np.abs(np.diff(x, axis=0)).mean(axis=0),  # how jerky
        (dyn ** 2).mean(axis=0),                  # motion energy
    ])
    coarse = w.reshape(-1, FS // 10, 3).mean(axis=1).ravel()  # 10 Hz shape of the gesture
    return np.concatenate([stats, coarse])


def load_augmented() -> tuple[np.ndarray, pd.DataFrame]:
    manifest_path = AUG_DIR / "manifest.csv"
    if not manifest_path.exists():
        return np.empty((0, WINDOW, 3)), pd.DataFrame()
    manifest = pd.read_csv(manifest_path)
    windows = np.stack([pd.read_csv(REPO_ROOT / f)[["ax", "ay", "az"]].to_numpy() for f in manifest["file"]])
    return windows, manifest


def choose_groups(meta: pd.DataFrame) -> tuple[str, pd.Series]:
    if meta["person"].nunique() >= 2:
        return "person", meta["person"]
    sessions = meta["person"] + "_s" + meta["session"].astype(str)
    if sessions.nunique() >= 2:
        return "session", sessions
    return "recording", meta["source"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-augmented", action="store_true", help="train on raw windows only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_w, meta_list = load_raw_windows()
    meta = pd.DataFrame(meta_list)
    X = np.stack([features(w) for w in raw_w])
    y = meta["gesture"].to_numpy()

    aug_w, aug_meta = (np.empty((0, WINDOW, 3)), pd.DataFrame()) if args.no_augmented else load_augmented()
    X_aug = np.stack([features(w) for w in aug_w]) if len(aug_w) else np.empty((0, X.shape[1]))
    print(f"raw windows: {len(X)}   augmented windows: {len(X_aug)}")
    print(meta.groupby(["gesture", "person"]).size().unstack(fill_value=0).to_string(), "\n")

    group_kind, groups = choose_groups(meta)
    print(f"cross-validation: leave-one-{group_kind}-out over {groups.nunique()} groups\n")

    y_true, y_pred = [], []
    for g in sorted(groups.unique()):
        test = (groups == g).to_numpy()
        train_sources = set(meta.loc[~test, "source"])
        X_train, y_train = X[~test], y[~test]
        if len(X_aug):
            keep = aug_meta["source"].isin(train_sources).to_numpy()
            X_train = np.vstack([X_train, X_aug[keep]])
            y_train = np.concatenate([y_train, aug_meta.loc[keep, "gesture"].to_numpy()])

        clf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X[test])
        y_true.extend(y[test])
        y_pred.extend(pred)
        print(f"  held out {g:<14} {test.sum():4d} windows   accuracy {accuracy_score(y[test], pred):.3f}")

    labels = [c for c in CLASSES if c in set(y_true)]
    print(f"\noverall held-out accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                      index=[f"true {c}" for c in labels], columns=[f"pred {c}" for c in labels])
    print(cm.to_string())

    X_all = np.vstack([X, X_aug]) if len(X_aug) else X
    y_all = np.concatenate([y, aug_meta["gesture"].to_numpy()]) if len(X_aug) else y
    final = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1).fit(X_all, y_all)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "classes": CLASSES, "fs": FS, "window": WINDOW}, MODEL_PATH)
    print(f"\nfinal model trained on all {len(X_all)} windows -> {Path(MODEL_PATH).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
