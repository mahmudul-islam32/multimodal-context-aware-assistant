"""Tokenisation shared by retriever and grounding check."""

from __future__ import annotations

import re

STOP = set(
    """a an the and or but if then else of to in on at by for with from into onto about as is are was were be been
    being do does did doing have has had having i me my we our you your he she it its they them their this that these
    those what which who whom when where why how can could should would may might must shall will not no yes there
    here please tell me any some all each every more most very just also than so up down out over under again only
    own same too s t don doesn didn isn aren wasn weren hasn haven hadn won wouldn shouldn couldn mustn let us get
    need want know""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def stem(w: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP and (len(t) > 2 or any(c.isdigit() for c in t))]


def identifiers(text: str) -> set[str]:
    """Model numbers, part IDs, doses: tokens with digits and letters, e.g. P-101, 0.9%."""
    return {t for t in _TOKEN.findall(text.lower()) if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)}


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]
