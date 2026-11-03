"""Follow-up question resolution (conversation memory).

Workers ask short follow-ups ("and at night?", "what about the guard
screws?", "how long for that?"). Retrieval on such a fragment fails or hits
the wrong section, so a follow-up is expanded with the content terms of the
previous question. The rule is deliberately simple and transparent: the
resolved query is shown in the answer and logged.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from .text import STOP, content_tokens

_FOLLOWUP_START = re.compile(r"^\s*(and|also|what about|how about|and what about|same for|then|so)\b", re.I)
_ANAPHORA = re.compile(r"\b(it|its|that|this|those|these|they|them|there|one)\b", re.I)
_RAW_WORD = re.compile(r"[A-Za-z0-9][\w.-]*")


@dataclass
class Resolution:
    retrieval_query: str  # what the retriever sees (own terms weighted x2)
    display: str  # what the user is shown ("Interpreted as ...")
    expanded: bool


@dataclass
class Turn:
    query: str
    resolved: str
    refused: bool


class ConversationMemory:
    def __init__(self, max_turns: int = 4, max_followup_terms: int = 4):
        self.turns: deque[Turn] = deque(maxlen=max_turns)
        self.max_followup_terms = max_followup_terms

    def clear(self) -> None:
        self.turns.clear()

    def is_followup(self, query: str) -> bool:
        if not self.turns:
            return False
        n = len(content_tokens(query))
        if _FOLLOWUP_START.search(query):
            return True
        return n <= self.max_followup_terms and bool(_ANAPHORA.search(query))

    def resolve(self, query: str) -> Resolution:
        """Expand a follow-up with the previous question's terms.

        The follow-up's own words are repeated so they outweigh the carried
        context when they name a new subject ("what about the guard screws?").
        """
        same = Resolution(query, query, False)
        if not self.is_followup(query):
            return same
        prev = next((t for t in reversed(self.turns) if not t.refused), None)
        if prev is None:
            return same
        mine = set(content_tokens(query))
        words = [w for w in _RAW_WORD.findall(prev.resolved) if w.lower() not in STOP and len(w) > 2]
        extra: list[str] = []
        for w in words:
            if content_tokens(w) and content_tokens(w)[0] not in mine and w.lower() not in (e.lower() for e in extra):
                extra.append(w)
        if not extra:
            return same
        cleaned = _FOLLOWUP_START.sub("", query).strip(" ?,.")
        ctx = " ".join(extra)
        retrieval = f"{cleaned} {cleaned} ({ctx})" if len(mine) >= 2 else f"{cleaned} ({ctx})"
        return Resolution(retrieval, f"{cleaned} ({ctx})?", True)

    def add(self, query: str, resolved: str, refused: bool) -> None:
        self.turns.append(Turn(query, resolved, refused))
