"""F-01: sensor streaming and signal preprocessing."""

from .preprocess import (
    CHANNELS,
    StreamBuffer,
    butter_lowpass,
    detect_drops,
    resample,
    separate_gravity,
    sliding_windows,
)

__all__ = [
    "CHANNELS",
    "StreamBuffer",
    "butter_lowpass",
    "detect_drops",
    "resample",
    "separate_gravity",
    "sliding_windows",
]
