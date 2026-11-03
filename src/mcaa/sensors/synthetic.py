"""Synthetic IMU generator (UCI-HAR-like, 50 Hz, acc in g, gyro in rad/s).

Used for CI, demos and figures when the public datasets are not available.
It models per-subject variation (gait frequency, amplitude, device tilt) so
that leave-subjects-out evaluation is meaningful, but it is **not** a
substitute for real data: numbers produced on it are labelled "synthetic".
"""

from __future__ import annotations

import numpy as np

ACTIVITIES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
    "WORKING",  # hands-busy manual work, added via calibration recordings
)

# activity -> task-relevant coarse state used by the context engine
ACTIVITY_TO_TASK = {
    "WALKING": "moving",
    "WALKING_UPSTAIRS": "moving",
    "WALKING_DOWNSTAIRS": "moving",
    "SITTING": "idle",
    "STANDING": "idle",
    "LAYING": "idle",
    "WORKING": "hands_busy",
}


def _rot(tilt: np.ndarray) -> np.ndarray:
    a, b, c = tilt
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


class SubjectProfile:
    def __init__(self, subject_id: int, seed: int = 0):
        rng = np.random.default_rng(seed * 1000 + subject_id)
        self.id = subject_id
        self.step_hz = rng.uniform(1.6, 2.1)
        self.amp = rng.uniform(0.75, 1.25)
        self.tilt = rng.normal(0, 0.12, 3)
        self.tremor = rng.uniform(0.004, 0.012)
        self.work_hz = rng.uniform(2.5, 4.5)
        self.rng = rng


def generate_segment(activity: str, seconds: float, subject: SubjectProfile, fs: float = 50.0) -> np.ndarray:
    """Return (n, 6) array [ax, ay, az, gx, gy, gz]."""
    rng = subject.rng
    n = int(seconds * fs)
    t = np.arange(n) / fs
    f = subject.step_hz * rng.uniform(0.95, 1.05)
    A = subject.amp
    acc = np.zeros((n, 3))
    gyr = np.zeros((n, 3))
    phase = rng.uniform(0, 2 * np.pi)
    if activity.startswith("WALKING"):
        k = {"WALKING": 1.0, "WALKING_UPSTAIRS": 0.8, "WALKING_DOWNSTAIRS": 1.25}[activity]
        fk = f * {"WALKING": 1.0, "WALKING_UPSTAIRS": 0.85, "WALKING_DOWNSTAIRS": 1.1}[activity]
        acc[:, 0] = 1.0 + 0.25 * A * k * np.sin(2 * np.pi * fk * t + phase) + 0.08 * A * k * np.sin(4 * np.pi * fk * t)
        acc[:, 1] = 0.12 * A * k * np.sin(np.pi * fk * t + phase)
        acc[:, 2] = 0.10 * A * k * np.sin(2 * np.pi * fk * t + phase + 1.0)
        if activity == "WALKING_UPSTAIRS":
            acc[:, 2] += 0.08
            gyr[:, 2] += 0.15 * np.sin(np.pi * fk * t)
        if activity == "WALKING_DOWNSTAIRS":
            acc[:, 2] -= 0.08
            acc[:, 0] += 0.12 * A * (np.sin(2 * np.pi * fk * t + phase) > 0.9)
        gyr[:, 0] = 0.5 * A * k * np.sin(np.pi * fk * t + phase)
        gyr[:, 1] = 0.35 * A * k * np.sin(2 * np.pi * fk * t + phase + 0.5)
        gyr[:, 2] += 0.25 * A * k * np.sin(np.pi * fk * t)
    elif activity == "SITTING":
        acc[:] = [0.85, 0.05, 0.5]
    elif activity == "STANDING":
        acc[:] = [1.0, 0.0, 0.05]
        acc += 0.01 * np.sin(2 * np.pi * 0.3 * t)[:, None]
    elif activity == "LAYING":
        acc[:] = [0.05, 0.1, 0.99]
    elif activity == "WORKING":
        # standing upper body, irregular high-frequency wrist/arm motion
        acc[:] = [0.95, 0.1, 0.2]
        bursts = (np.sin(2 * np.pi * 0.4 * t + phase) > -0.2).astype(float)
        wf = subject.work_hz
        acc[:, 1] += 0.15 * A * bursts * np.sin(2 * np.pi * wf * t)
        acc[:, 2] += 0.10 * A * bursts * np.sin(2 * np.pi * wf * 1.3 * t + 0.7)
        gyr[:, 0] = 0.9 * A * bursts * np.sin(2 * np.pi * wf * t + 0.3)
        gyr[:, 1] = 0.6 * A * bursts * np.sin(2 * np.pi * wf * 0.7 * t)
        gyr[:, 2] = 0.4 * A * bursts * rng.normal(0, 1, n)
    else:
        raise ValueError(f"unknown activity {activity}")
    # sensor noise and physiological tremor
    acc += rng.normal(0, subject.tremor + 0.006, acc.shape)
    gyr += rng.normal(0, 0.01 + subject.tremor, gyr.shape)
    R = _rot(subject.tilt)
    acc = acc @ R.T
    gyr = gyr @ R.T
    return np.hstack([acc, gyr])


def make_dataset(
    n_subjects: int = 12,
    seconds_per_activity: float = 60.0,
    activities: tuple[str, ...] = ACTIVITIES,
    seed: int = 42,
    fs: float = 50.0,
    window: int = 128,
    step: int = 64,
):
    """Windows X (n, window, 6), labels y (str), subject ids g."""
    from .preprocess import preprocess_block, sliding_windows

    X, y, g = [], [], []
    for s in range(1, n_subjects + 1):
        prof = SubjectProfile(s, seed)
        for act in activities:
            seg = preprocess_block(generate_segment(act, seconds_per_activity, prof, fs), fs)
            w = sliding_windows(seg, window, step)
            X.append(w)
            y += [act] * len(w)
            g += [s] * len(w)
    return np.concatenate(X), np.array(y), np.array(g)


def scripted_session(script: list[tuple[str, float]], subject_id: int = 99, seed: int = 7, fs: float = 50.0):
    """Continuous recording following a script of (activity, seconds)."""
    prof = SubjectProfile(subject_id, seed)
    parts, labels = [], []
    for act, sec in script:
        seg = generate_segment(act, sec, prof, fs)
        parts.append(seg)
        labels += [act] * len(seg)
    x = np.concatenate(parts)
    t = np.arange(len(x)) / fs
    return t, x, np.array(labels)
