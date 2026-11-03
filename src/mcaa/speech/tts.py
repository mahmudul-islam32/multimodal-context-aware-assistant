"""Text-to-speech with interruptible (barge-in) playback.

In the web UI speech is synthesised in the browser (Web Speech API) and
barge-in is handled there. Server-side, :class:`PiperTTS` wraps the Piper CLI
if installed and :class:`PlaybackController` provides stop-on-barge-in for a
local audio sink.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class PiperTTS:
    name = "piper"

    def __init__(self, voice: str = "en_US-lessac-medium"):
        if not shutil.which("piper"):
            raise RuntimeError("piper binary not found")
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        p = subprocess.run(
            ["piper", "--model", self.voice, "--output_file", "-"],
            input=text.encode(), capture_output=True, check=True, timeout=30,
        )
        return p.stdout


class BrowserTTS:
    """Marker backend: the client speaks the text itself."""

    name = "browser"

    def synthesize(self, text: str) -> bytes:
        return b""


def get_tts(preference: str = "auto"):
    if preference in ("auto", "piper"):
        try:
            return PiperTTS()
        except RuntimeError:
            pass
    return BrowserTTS()


@dataclass
class PlaybackController:
    """Plays text chunk by chunk; ``barge_in()`` stops at the next chunk boundary.

    ``sink`` receives each chunk (words) and should block for its duration.
    """

    sink: Callable[[str], None] = field(default=lambda chunk: time.sleep(0.05 * len(chunk.split())))
    chunk_words: int = 4
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    started_at: float | None = None
    spoken: list[str] = field(default_factory=list)
    interrupted: bool = False

    def play(self, text: str, ready_at: float | None = None) -> None:
        self._stop.clear()
        self.spoken, self.interrupted = [], False
        words = text.split()

        def run():
            self.started_at = time.perf_counter()
            for i in range(0, len(words), self.chunk_words):
                if self._stop.is_set():
                    self.interrupted = True
                    return
                chunk = " ".join(words[i : i + self.chunk_words])
                self.sink(chunk)
                self.spoken.append(chunk)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def barge_in(self) -> None:
        """User tapped 'stop' or said 'stop'."""
        self._stop.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def playing(self) -> bool:
        return bool(self._thread and self._thread.is_alive())


STOP_WORDS = {"stop", "cancel", "quiet", "be quiet", "enough"}


def is_barge_in_command(transcript: str) -> bool:
    return transcript.strip().lower().strip(".!") in STOP_WORDS
