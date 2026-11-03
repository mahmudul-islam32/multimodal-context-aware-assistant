"""Choose modality, length and detail from the context state (versioned YAML)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ..config import PolicyConfig, StatePolicy, load_policy_config


@dataclass
class OutputPlan:
    policy_version: str
    state: str
    modality: str
    speak: bool
    voice_text: str
    screen_text: str
    screen_detail: str
    visual_cue: bool
    deferred: bool
    citations: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def truncate_words(text: str, n: int) -> str:
    """First sentence(s) within n words; hard cut with ellipsis otherwise."""
    if n <= 0:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for s in sents:
        if len((" ".join(out + [s])).split()) <= n:
            out.append(s)
        else:
            break
    if out:
        return " ".join(out)
    words = text.split()
    return " ".join(words[:n]).rstrip(",;:") + "…"


def _strip_markers(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text)


class OutputPolicy:
    def __init__(self, config: PolicyConfig | None = None):
        self.cfg = config or load_policy_config()

    def rules_for(self, state: str, plain: bool) -> StatePolicy:
        return self.cfg.plain_chatbot if plain else self.cfg.states[state]

    def plan(self, state: str, answer: str, short_answer: str, citations: list[dict], refused: bool, plain: bool = False) -> OutputPlan:
        r = self.rules_for(state, plain)
        spoken_src = short_answer or answer
        speak = r.voice != "off" and r.max_voice_words > 0 and r.modality != "screen"
        voice = _strip_markers(truncate_words(spoken_src, r.max_voice_words)) if speak else ""
        if speak and citations and r.citations == "short" and not refused:
            src = citations[0]
            tag = f" From the {src['title']}, {src['section']}."
            if len((voice + tag).split()) <= r.max_voice_words + 8:
                voice += tag
        screen = short_answer if r.screen_detail == "summary" and short_answer else answer
        return OutputPlan(
            policy_version=self.cfg.version,
            state="PLAIN" if plain else state,
            modality=r.modality,
            speak=speak,
            voice_text=voice,
            screen_text=screen,
            screen_detail=r.screen_detail,
            visual_cue=r.visual_cue,
            deferred=r.defer_details and screen != answer,
            citations=citations,
        )
