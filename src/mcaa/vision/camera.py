"""Lightweight on-device camera cues.

Privacy by design: disabled until explicit consent; frames are processed in
memory and never written anywhere. Only a 32x32 grey thumbnail of the last
frame is held for motion estimation and is dropped on ``disable()``.

Cues:
- ``hands_in_view``: share of skin-tone pixels (YCrCb rule) in the lower
  half of the frame, i.e. hands working in front of the camera.
- ``motion``: mean absolute difference between consecutive thumbnails.
- ``hands_busy``: combination of both, fed to the context engine.

A heavier detector (e.g. hand/object detection for component recognition)
can be plugged in via ``detector`` - any callable frame -> dict of cues.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class ConsentRequired(PermissionError):
    pass


def _thumb(gray: np.ndarray, size: int = 32) -> np.ndarray:
    h, w = gray.shape
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return gray[np.ix_(ys, xs)].astype(np.float32)


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    return (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)


@dataclass
class CameraCues:
    hands_in_view: float
    motion: float
    hands_busy: float
    extra: dict


class CameraContext:
    def __init__(self, detector: Callable[[np.ndarray], dict] | None = None):
        self.enabled = False
        self._prev: np.ndarray | None = None
        self.detector = detector
        self.frames_processed = 0

    def enable(self, consent: bool) -> None:
        if not consent:
            raise ConsentRequired("camera context requires explicit consent")
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False
        self._prev = None

    def process(self, frame: np.ndarray) -> CameraCues:
        if not self.enabled:
            raise ConsentRequired("camera context is disabled")
        rgb = frame[..., :3]
        gray = rgb.mean(-1)
        lower = rgb[rgb.shape[0] // 2 :]
        hands = float(skin_mask(lower).mean())
        th = _thumb(gray)
        motion = 0.0 if self._prev is None else float(np.abs(th - self._prev).mean() / 255)
        self._prev = th
        self.frames_processed += 1
        busy = float(np.clip(hands * 4, 0, 1) * np.clip(0.4 + motion * 20, 0, 1))
        extra = self.detector(frame) if self.detector else {}
        return CameraCues(round(hands, 4), round(motion, 4), round(busy, 4), extra)
