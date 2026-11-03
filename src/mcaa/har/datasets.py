"""Loaders for public HAR datasets (download with ``python -m mcaa.data.download``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..sensors.preprocess import preprocess_block, resample, sliding_windows

UCI_LABELS = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS", 4: "SITTING", 5: "STANDING", 6: "LAYING"}


def load_uci_har(root: Path):
    """UCI HAR raw inertial signals: (n, 128, 6) total_acc[g] + body_gyro[rad/s]."""
    base = next(Path(root).rglob("UCI HAR Dataset"), Path(root))
    Xs, ys, gs = [], [], []
    for split in ("train", "test"):
        d = base / split / "Inertial Signals"
        chans = [f"total_acc_{a}" for a in "xyz"] + [f"body_gyro_{a}" for a in "xyz"]
        X = np.stack([np.loadtxt(d / f"{c}_{split}.txt") for c in chans], axis=-1)
        y = np.loadtxt(base / split / f"y_{split}.txt", dtype=int)
        g = np.loadtxt(base / split / f"subject_{split}.txt", dtype=int)
        Xs.append(X)
        ys.append(np.array([UCI_LABELS[i] for i in y]))
        gs.append(g)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


WISDM_CODES = {
    "A": "WALKING", "B": "JOGGING", "C": "WALKING_STAIRS", "D": "SITTING", "E": "STANDING",
    "F": "TYPING", "G": "BRUSHING_TEETH", "H": "EATING_SOUP", "I": "EATING_CHIPS", "J": "EATING_PASTA",
    "K": "DRINKING", "L": "EATING_SANDWICH", "M": "KICKING", "O": "CATCHING", "P": "DRIBBLING",
    "Q": "WRITING", "R": "CLAPPING", "S": "FOLDING",
}


def _read_wisdm(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.strip().rstrip(";").split(",")
        if len(parts) == 6:
            rows.append((int(parts[0]), parts[1], int(parts[2]), *map(float, parts[3:])))
    return rows


def load_wisdm(root: Path, device: str = "watch"):
    """WISDM 2019 (20 Hz): resampled to 50 Hz, acc converted from m/s^2 to g."""
    base = next(Path(root).rglob("wisdm-dataset"), Path(root))
    acc_dir, gyr_dir = base / "raw" / device / "accel", base / "raw" / device / "gyro"
    Xs, ys, gs = [], [], []
    for af in sorted(acc_dir.glob("*.txt")):
        gf = gyr_dir / af.name.replace("accel", "gyro")
        if not gf.exists():
            continue
        acc, gyr = _read_wisdm(af), _read_wisdm(gf)
        for code, label in WISDM_CODES.items():
            a = np.array([r[2:] for r in acc if r[1] == code], dtype=float)
            g_ = np.array([r[2:] for r in gyr if r[1] == code], dtype=float)
            if len(a) < 100 or len(g_) < 100:
                continue
            ta, xa = resample(a[:, 0] / 1e9, a[:, 1:] / 9.81)
            tg, xg = resample(g_[:, 0] / 1e9, g_[:, 1:])
            n = min(len(xa), len(xg))
            w = sliding_windows(preprocess_block(np.hstack([xa[:n], xg[:n]])))
            subj = int(acc[0][0])
            Xs.append(w)
            ys += [label] * len(w)
            gs += [subj] * len(w)
    return np.concatenate(Xs), np.array(ys), np.array(gs)
