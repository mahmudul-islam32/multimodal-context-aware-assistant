import time

import numpy as np
import pytest

from mcaa.speech import EnergyVAD, PlaybackController, noise_level_dbfs, word_error_rate
from mcaa.speech.tts import is_barge_in_command
from mcaa.vision import CameraContext, ConsentRequired


def test_wer():
    assert word_error_rate("tighten the bolts", "tighten the bolts") == 0
    assert word_error_rate("tighten the bolts", "tighten bolts") == pytest.approx(1 / 3)


def test_vad_finds_speech_segment():
    sr = 16000
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.003, sr * 3)
    a[sr : 2 * sr] += 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    segs = EnergyVAD(sr).segments(a)
    assert len(segs) == 1 and 0.9 < segs[0][0] < 1.1 and 1.9 < segs[0][1] < 2.1
    assert noise_level_dbfs(a[:sr]) < -40


def test_barge_in_stops_playback():
    pc = PlaybackController(sink=lambda chunk: time.sleep(0.05), chunk_words=1)
    pc.play("one two three four five six seven eight nine ten")
    time.sleep(0.12)
    pc.barge_in()
    pc.wait(1)
    assert pc.interrupted and len(pc.spoken) < 10
    assert is_barge_in_command("Stop.") and not is_barge_in_command("stop the pump how?")


def test_camera_requires_consent_and_detects_hands():
    cam = CameraContext()
    frame = np.zeros((120, 160, 3), np.uint8)
    with pytest.raises(ConsentRequired):
        cam.process(frame)
    with pytest.raises(ConsentRequired):
        cam.enable(False)
    cam.enable(True)
    cam.process(frame)
    skin = frame.copy()
    skin[60:, 40:120] = (205, 150, 125)
    cues = cam.process(skin)
    assert cues.hands_in_view > 0.3 and cues.motion > 0 and cues.hands_busy > 0.5
    cam.disable()
    assert cam._prev is None
