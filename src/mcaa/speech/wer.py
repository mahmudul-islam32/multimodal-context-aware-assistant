"""Word error rate (Levenshtein over normalised words)."""

from __future__ import annotations

import re


def normalise(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    r, h = normalise(reference), normalise(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    errs = sum(word_error_rate(r, h) * len(normalise(r)) for r, h in pairs)
    return errs / max(1, sum(len(normalise(r)) for r, _ in pairs))
