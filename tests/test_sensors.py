import numpy as np

from mcaa.sensors import StreamBuffer, detect_drops, resample, sliding_windows
from mcaa.sensors.synthetic import SubjectProfile, generate_segment


def test_resample_20hz_to_50hz():
    t = np.arange(0, 10, 1 / 20)
    x = np.sin(2 * np.pi * 1.0 * t)
    tr, xr = resample(t, x, 50)
    assert abs(np.median(np.diff(tr)) - 0.02) < 1e-9
    assert np.abs(xr[:, 0] - np.sin(2 * np.pi * tr)).max() < 0.02


def test_detect_drops():
    t = np.delete(np.arange(0, 4, 0.02), np.r_[50:55])
    gaps = detect_drops(t, 50)
    assert len(gaps) == 1 and gaps[0][2] == 5


def test_windows_shape():
    w = sliding_windows(np.zeros((512, 6)))
    assert w.shape == (7, 128, 6)


def test_stream_buffer_emits_windows_and_counts_drops():
    seg = generate_segment("WALKING", 8, SubjectProfile(1))
    buf = StreamBuffer()
    out = []
    for i, row in enumerate(seg):
        if 100 <= i < 104:
            continue  # simulate dropped packets
        w = buf.push({"t": i / 50, **dict(zip(("ax", "ay", "az", "gx", "gy", "gz"), row, strict=False))})
        if w is not None:
            out.append(w)
    assert out and out[0].shape == (128, 6)
    assert buf.stats.dropped == 4 and buf.stats.gaps == 1
