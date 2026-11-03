"""Energy-based voice activity detection and ambient noise estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def noise_level_dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS for float audio in [-1, 1]."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    return 20 * np.log10(max(rms, 1e-10))


@dataclass
class EnergyVAD:
    sample_rate: int = 16000
    frame_ms: int = 30
    threshold_db: float = 12.0  # speech must exceed noise floor by this many dB
    min_silence_ms: int = 450

    def frames(self, audio: np.ndarray) -> np.ndarray:
        n = int(self.sample_rate * self.frame_ms / 1000)
        k = len(audio) // n
        return audio[: k * n].reshape(k, n) if k else np.empty((0, n))

    def speech_mask(self, audio: np.ndarray) -> np.ndarray:
        fr = self.frames(audio)
        if not len(fr):
            return np.zeros(0, bool)
        db = 20 * np.log10(np.sqrt((fr**2).mean(1)) + 1e-10)
        floor = np.percentile(db, 10)
        return db > floor + self.threshold_db

    def segments(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """Speech segments in seconds, merged across short pauses."""
        m = self.speech_mask(audio)
        hop = self.frame_ms / 1000
        gap = int(self.min_silence_ms / self.frame_ms)
        segs, start, silent = [], None, 0
        for i, s in enumerate(m):
            if s:
                start = i if start is None else start
                silent = 0
            elif start is not None:
                silent += 1
                if silent > gap:
                    segs.append((start * hop, (i - silent + 1) * hop))
                    start, silent = None, 0
        if start is not None:
            segs.append((start * hop, (len(m) - silent) * hop))
        return segs
