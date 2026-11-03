"""Hand-crafted time and frequency domain features per window (UCI-HAR style)."""

from __future__ import annotations

import numpy as np

from ..sensors.preprocess import separate_gravity

_STATS = ("mean", "std", "mad", "min", "max", "iqr", "energy", "entropy", "dom_freq", "band_lo", "band_hi")


def _entropy(x: np.ndarray, bins: int = 16) -> float:
    hist, _ = np.histogram(x, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def _channel_feats(x: np.ndarray, fs: float) -> list[float]:
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    tot = spec.sum() + 1e-12
    q75, q25 = np.percentile(x, [75, 25])
    return [
        x.mean(),
        x.std(),
        np.median(np.abs(x - np.median(x))),
        x.min(),
        x.max(),
        q75 - q25,
        float((x**2).mean()),
        _entropy(x),
        float(freqs[np.argmax(spec)]),
        float(spec[(freqs > 0.3) & (freqs <= 2.5)].sum() / tot),
        float(spec[freqs > 2.5].sum() / tot),
    ]


def _signals(w: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    acc, gyr = w[:, :3], w[:, 3:6]
    body, grav = separate_gravity(acc, fs)
    sig = {}
    for i, a in enumerate("xyz"):
        sig[f"body_acc_{a}"] = body[:, i]
        sig[f"grav_{a}"] = grav[:, i]
        sig[f"gyro_{a}"] = gyr[:, i]
    sig["body_acc_mag"] = np.linalg.norm(body, axis=1)
    sig["gyro_mag"] = np.linalg.norm(gyr, axis=1)
    sig["jerk_mag"] = np.r_[0, np.linalg.norm(np.diff(body, axis=0), axis=1) * fs]
    return sig


def feature_names(fs: float = 50.0) -> list[str]:
    dummy = _signals(np.zeros((128, 6)), fs)
    names = [f"{k}_{s}" for k in dummy for s in _STATS]
    names += ["corr_acc_xy", "corr_acc_xz", "corr_acc_yz", "corr_gyro_xy"]
    return names


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(windows: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """(n, T, 6) -> (n, n_features)."""
    if windows.ndim == 2:
        windows = windows[None]
    out = []
    for w in windows:
        sig = _signals(w, fs)
        f = []
        for v in sig.values():
            f += _channel_feats(v, fs)
        b = w[:, :3] - w[:, :3].mean(0)
        f += [_corr(b[:, 0], b[:, 1]), _corr(b[:, 0], b[:, 2]), _corr(b[:, 1], b[:, 2]), _corr(w[:, 3], w[:, 4])]
        out.append(f)
    return np.nan_to_num(np.asarray(out, dtype=np.float32))
