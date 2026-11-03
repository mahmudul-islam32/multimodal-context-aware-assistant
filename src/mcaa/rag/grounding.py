"""Claim-level grounding check: every sentence must be supported by a cited chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import content_tokens, numbers


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[int]
    support: float
    supported: bool
    reason: str = ""


@dataclass
class GroundingReport:
    checks: list[SentenceCheck] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        return sum(c.supported for c in self.checks) / len(self.checks) if self.checks else 0.0

    @property
    def supported_sentences(self) -> list[SentenceCheck]:
        return [c for c in self.checks if c.supported]


_CITE = re.compile(r"\[(\d+)\]")


def split_cited(answer: str) -> list[tuple[str, list[int]]]:
    """'A [1]. B [2][3].' -> [('A.', [1]), ('B.', [2, 3])]"""
    parts = re.split(r"(?<=[.!?\]])\s+(?=[A-Z0-9])", answer.strip())
    out = []
    for p in parts:
        cites = [int(c) for c in _CITE.findall(p)]
        text = _CITE.sub("", p).strip()
        if text:
            out.append((re.sub(r"\s+([.,;:!?])", r"\1", text), cites))
    return out


def check(answer: str, sources: dict[int, str], threshold: float = 0.6) -> GroundingReport:
    rep = GroundingReport()
    for sent, cites in split_cited(answer):
        if not cites:
            rep.checks.append(SentenceCheck(sent, [], 0.0, False, "no citation"))
            continue
        toks = set(content_tokens(sent))
        best, reason = 0.0, ""
        for c in cites:
            src = sources.get(c)
            if src is None:
                reason = f"citation [{c}] does not exist"
                continue
            stoks = set(content_tokens(src))
            sup = len(toks & stoks) / len(toks) if toks else 1.0
            missing_nums = numbers(sent) - numbers(src)
            if missing_nums:
                sup = min(sup, 0.0)
                reason = f"numbers not in source: {sorted(missing_nums)}"
            best = max(best, sup)
        rep.checks.append(SentenceCheck(sent, cites, round(best, 3), best >= threshold, reason if best < threshold else ""))
    return rep
