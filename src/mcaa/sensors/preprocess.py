"""Resampling, filtering, windowing and drop detection for IMU streams.

Conventions follow UCI HAR: 50 Hz, windows of 128 samples (2.56 s) with 50 %
overlap, 3rd-order Butterworth low-pass at 20 Hz for noise and 0.3 Hz to split
gravity from body acceleration.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
TARGET_HZ = 50.0
WINDOW = 128
STEP = 64


def butter_lowpass(x: np.ndarray, cutoff: float, fs: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0."""
    nyq = 0.5 * fs
    cutoff = min(cutoff, 0.99 * nyq)
    b, a = signal.butter(order, cutoff / nyq, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        return x.copy()
    return signal.filtfilt(b, a, x, axis=0)


def separate_gravity(acc: np.ndarray, fs: float = TARGET_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Split total acceleration into (body, gravity) with a 0.3 Hz low-pass."""
    gravity = butter_lowpass(acc, 0.3, fs)
    return acc - gravity, gravity


def resample(t: np.ndarray, x: np.ndarray, fs_out: float = TARGET_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Resample irregularly timestamped samples onto a uniform grid.

    Works for 20 Hz (WISDM) and 50 Hz (UCI HAR) sources as well as jittery
    phone streams. ``t`` in seconds, ``x`` shape (n, channels).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    order = np.argsort(t)
    t, x = t[order], x[order]
    t, idx = np.unique(t, return_index=True)
    x = x[idx]
    if len(t) < 2:
        return t, x
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs_out)
    out = np.column_stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])])
    return grid, out


def detect_drops(t: np.ndarray, fs: float, tolerance: float = 1.5) -> list[tuple[float, float, int]]:
    """Return gaps as (start_time, end_time, n_missing) where dt > tolerance/fs."""
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return []
    dt = np.diff(t)
    gaps = np.where(dt > tolerance / fs)[0]
    return [(float(t[i]), float(t[i + 1]), int(round(dt[i] * fs)) - 1) for i in gaps]


def sliding_windows(x: np.ndarray, size: int = WINDOW, step: int = STEP) -> np.ndarray:
    """Cut (n, c) into (n_windows, size, c)."""
    if x.shape[0] < size:
        return np.empty((0, size, x.shape[1]))
    starts = range(0, x.shape[0] - size + 1, step)
    return np.stack([x[s : s + size] for s in starts])


def preprocess_block(x: np.ndarray, fs: float = TARGET_HZ) -> np.ndarray:
    """Noise filter (20 Hz) for a (n, 6) block of acc+gyro."""
    return butter_lowpass(x, 20.0, fs)


@dataclass
class StreamStats:
    received: int = 0
    dropped: int = 0
    gaps: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = np.array(self.latencies_ms[-500:]) if self.latencies_ms else np.array([0.0])
        return {
            "received": self.received,
            "dropped": self.dropped,
            "gaps": self.gaps,
            "drop_rate": self.dropped / max(1, self.received + self.dropped),
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
        }


class StreamBuffer:
    """Accumulates live IMU samples and emits preprocessed windows.

    Samples are dicts ``{"t": seconds, "ax": .., ..., "gz": ..}`` (as sent by
    the phone page over WebSocket). Optionally ``"sent_at"`` (epoch seconds)
    for latency measurement.
    """

    def __init__(self, source_hz: float = TARGET_HZ, window: int = WINDOW, step: int = STEP):
        self.source_hz = source_hz
        self.window = window
        self.step = step
        self.stats = StreamStats()
        self._t: deque[float] = deque(maxlen=window * 8)
        self._x: deque[list[float]] = deque(maxlen=window * 8)
        self._since_emit = 0
        self._last_t: float | None = None

    def push(self, sample: dict) -> np.ndarray | None:
        t = float(sample["t"])
        if "sent_at" in sample:
            self.stats.latencies_ms.append(max(0.0, (time.time() - float(sample["sent_at"])) * 1000))
        if self._last_t is not None:
            if t <= self._last_t:
                return None  # out-of-order / duplicate
            missing = int(round((t - self._last_t) * self.source_hz)) - 1
            if missing > 0 and (t - self._last_t) > 1.5 / self.source_hz:
                self.stats.dropped += missing
                self.stats.gaps += 1
        self._last_t = t
        self.stats.received += 1
        self._t.append(t)
        self._x.append([float(sample.get(c, 0.0)) for c in CHANNELS])
        self._since_emit += 1
        # emit roughly every `step` target samples
        needed = self.step * self.source_hz / TARGET_HZ
        if self._since_emit >= needed and len(self._t) >= 2:
            span = self._t[-1] - self._t[0]
            if span >= (self.window - 1) / TARGET_HZ:
                self._since_emit = 0
                return self.latest_window()
        return None

    def latest_window(self) -> np.ndarray | None:
        t = np.array(self._t)
        x = np.array(self._x)
        grid, xr = resample(t, x, TARGET_HZ)
        if len(grid) < self.window:
            return None
        xr = preprocess_block(xr)
        return xr[-self.window :]
