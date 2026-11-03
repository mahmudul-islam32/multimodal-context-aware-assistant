"""Speech-to-text backends. Audio is processed locally and never persisted here.

Backends:
- ``faster-whisper`` (default if installed): CTranslate2, CPU int8.
- ``whisper``: openai-whisper reference implementation.
- ``browser``: transcript already produced client-side (Web Speech API);
  the server only receives text plus a confidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .vad import EnergyVAD, noise_level_dbfs


@dataclass
class ASRResult:
    text: str
    confidence: float
    latency_ms: float
    noise_dbfs: float | None = None
    backend: str = ""


class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        from faster_whisper import WhisperModel  # type: ignore

        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        t0 = time.perf_counter()
        segs = EnergyVAD(sample_rate).segments(audio)
        clip = audio[int(segs[0][0] * sample_rate) : int(segs[-1][1] * sample_rate) + 1] if segs else audio
        out, _ = self.model.transcribe(clip.astype(np.float32), language="en", beam_size=1)
        out = list(out)
        text = " ".join(s.text.strip() for s in out)
        conf = float(np.exp(np.mean([s.avg_logprob for s in out]))) if out else 0.0
        return ASRResult(text, conf, (time.perf_counter() - t0) * 1000, noise_level_dbfs(audio), self.name)


class WhisperASR:
    name = "whisper"

    def __init__(self, model_size: str = "base.en"):
        import whisper  # type: ignore

        self.model = whisper.load_model(model_size)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        t0 = time.perf_counter()
        r = self.model.transcribe(audio.astype(np.float32), language="en", fp16=False)
        segs = r.get("segments", [])
        conf = float(np.exp(np.mean([s["avg_logprob"] for s in segs]))) if segs else 0.0
        return ASRResult(r["text"].strip(), conf, (time.perf_counter() - t0) * 1000, noise_level_dbfs(audio), self.name)


class UnavailableASR:
    name = "none"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        raise RuntimeError("No local ASR backend installed: pip install '.[speech]' (faster-whisper)")


def get_asr(preference: str = "auto", model_size: str = "base.en"):
    order = ["faster-whisper", "whisper"] if preference == "auto" else [preference]
    for name in order:
        try:
            return FasterWhisperASR(model_size) if name == "faster-whisper" else WhisperASR(model_size)
        except Exception:  # noqa: BLE001 - backend missing or model download failed
            continue
    return UnavailableASR()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes to mono float32."""
    import io
    import wave

    with wave.open(io.BytesIO(data)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(1)
    if sr != 16000:
        t = np.arange(len(raw)) / sr
        raw = np.interp(np.arange(0, t[-1], 1 / 16000), t, raw).astype(np.float32)
        sr = 16000
    return raw, sr
