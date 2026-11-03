"""Guarded RAG pipeline: input guardrails -> retrieval -> refusal gate ->
generation -> grounding check -> output check."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from ..config import Escalation
from .generator import ExtractiveGenerator, get_generator
from .grounding import check as grounding_check
from .guardrails import InjectionDetector, OutputChecker
from .pii import redact
from .retriever import Hit, HybridRetriever
from .store import DocumentStore
from .text import content_tokens, identifiers, numbers


@dataclass
class Answer:
    query: str
    redacted_query: str
    answer: str
    short_answer: str
    citations: list[dict]
    refused: bool
    refusal_reason: str = ""
    blocked: bool = False
    faithfulness: float = 0.0
    retrieval_confidence: float = 0.0
    generator: str = ""
    latency_ms: float = 0.0
    pii_found: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Thresholds:
    min_coverage: float = 0.5  # share of query content terms in best chunk
    min_bm25: float = 2.0
    grounding: float = 0.6
    top_k: int = 4


class RAGPipeline:
    def __init__(self, store: DocumentStore, escalation: Escalation, generator=None, thresholds: Thresholds | None = None):
        self.store = store
        self.escalation = escalation
        self.generator = generator or get_generator()
        self.th = thresholds or Thresholds()
        self.detector = store.detector if store else InjectionDetector()
        self.out_checker = OutputChecker()
        self.reindex()

    def reindex(self) -> None:
        self.retriever = HybridRetriever(self.store.active_chunks)

    # ------------------------------------------------------------------
    def refusal_text(self, why: str) -> str:
        return (
            f"I can't confirm this from the approved documents ({why}). "
            f"Please ask {self.escalation.name} ({self.escalation.contact})."
        )

    def _gate(self, query: str, hits: list[Hit], focus: str | None = None) -> tuple[bool, str, float]:
        if not hits:
            return False, "no matching document", 0.0
        best = hits[0]
        if focus:
            # follow-up: evidence must cover the user's own new terms
            ft = set(content_tokens(focus))
            if len(ft) >= 2:
                best_cov = max(len(ft & set(content_tokens(f"{h.chunk.title} {h.chunk.section} {h.chunk.text}"))) / len(ft) for h in hits)
                best = Hit(best.chunk, best.score, best.bm25, best.dense, max(best.coverage, best_cov))
        conf = min(1.0, 0.5 * best.coverage + 0.5 * min(1.0, best.bm25 / (3 * self.th.min_bm25)))
        ids = identifiers(query)
        if ids:
            have = set()
            for h in hits:
                have |= identifiers(f"{h.chunk.title} {h.chunk.text}")
            missing = ids - have
            if missing:
                return False, f"no approved document covers {', '.join(sorted(missing)).upper()}", conf * 0.3
        if best.coverage < self.th.min_coverage or hits[0].bm25 < self.th.min_bm25:
            return False, "evidence too weak", conf
        conflict = self._conflict(query, hits)
        if conflict:
            return False, conflict, conf
        return True, "", conf

    def _conflict(self, query: str, hits: list[Hit]) -> str:
        """Two strong chunks from different documents giving different numbers
        for the same query terms -> sources conflict."""
        strong = [h for h in hits[:3] if h.coverage >= self.th.min_coverage and h.bm25 >= self.th.min_bm25]
        q = set(content_tokens(query))
        for i, a in enumerate(strong):
            for b in strong[i + 1 :]:
                if a.chunk.doc_id == b.chunk.doc_id:
                    continue
                na = _numbers_near(a.chunk.text, q)
                nb = _numbers_near(b.chunk.text, q)
                if na and nb and not (na & nb):
                    return f"sources conflict: {a.chunk.title} vs {b.chunk.title}"
        return ""

    # ------------------------------------------------------------------
    def ask(self, query: str, step_hint: str = "", focus: str | None = None) -> Answer:
        """Answer ``query``. ``focus`` is the user's literal follow-up when
        ``query`` was expanded with conversation context."""
        t0 = time.perf_counter()
        events: list[dict] = []
        red, pii = redact(query)
        if pii:
            events.append({"type": "pii_redacted", "entities": pii})
        scan = self.detector.scan(query)
        if scan.flagged:
            events.append({"type": "injection_blocked", "where": "query", "score": scan.score, "matches": scan.matches})
            return Answer(query, red, "This request was blocked by the input safety check. I can only answer questions about the approved documents.",
                          "Request blocked by the safety check.", [], True, "input injection detected", True,
                          generator="guardrail", latency_ms=_ms(t0), pii_found=pii, events=events)

        hits = self.retriever.search(red + " " + step_hint if step_hint and len(content_tokens(red)) < 3 else red, k=self.th.top_k)
        ok, why, conf = self._gate(red, hits, redact(focus)[0] if focus else None)
        if not ok:
            events.append({"type": "refusal", "reason": why})
            txt = self.refusal_text(why)
            return Answer(query, red, txt, txt, [], True, why, retrieval_confidence=conf, generator="gate",
                          latency_ms=_ms(t0), pii_found=pii, events=events)

        hits = [h for h in hits if h.coverage >= 0.25] or hits[:1]
        try:
            draft = self.generator.generate(red, hits, step_hint)
        except Exception as e:  # noqa: BLE001 - LLM unreachable -> safe fallback
            events.append({"type": "generator_error", "error": str(e)[:200]})
            draft = ExtractiveGenerator().generate(red, hits, step_hint)
        sources = {i + 1: h.chunk.text for i, h in enumerate(hits)}
        if draft.refuse or not draft.answer:
            why = "generator found no supported answer"
            events.append({"type": "refusal", "reason": why})
            txt = self.refusal_text(why)
            return Answer(query, red, txt, txt, [], True, why, retrieval_confidence=conf, generator=draft.model,
                          latency_ms=_ms(t0), pii_found=pii, events=events)

        rep = grounding_check(draft.answer, sources, self.th.grounding)
        unsupported = [c.sentence for c in rep.checks if not c.supported]
        if unsupported:
            events.append({"type": "ungrounded_removed", "sentences": unsupported})
        kept = rep.supported_sentences
        if not kept:
            why = "answer not supported by the documents"
            txt = self.refusal_text(why)
            return Answer(query, red, txt, txt, [], True, why, faithfulness=rep.faithfulness, retrieval_confidence=conf,
                          generator=draft.model, latency_ms=_ms(t0), pii_found=pii, events=events, unsupported=unsupported)
        answer = " ".join(f"{c.sentence} " + "".join(f"[{i}]" for i in c.citations) for c in kept)
        short_rep = grounding_check(draft.short_answer, sources, self.th.grounding) if draft.short_answer else None
        short = draft.short_answer if short_rep and short_rep.faithfulness == 1.0 else f"{kept[0].sentence} [{kept[0].citations[0]}]"

        oc = self.out_checker.check(answer, list(sources.values()))
        if not oc.ok:
            events.append({"type": "output_blocked", "reasons": oc.reasons})
            txt = self.refusal_text("output safety check")
            return Answer(query, red, txt, txt, [], True, "output blocked", True, rep.faithfulness, conf, draft.model,
                          _ms(t0), pii, events)

        used = sorted({i for c in kept for i in c.citations} | set(_cites(short)))
        remap = {old: new + 1 for new, old in enumerate(used)}
        answer, short = _renumber(answer, remap), _renumber(short, remap)
        citations = [{**hits[i - 1].chunk.citation(), "n": remap[i], "text": hits[i - 1].chunk.text} for i in used]
        return Answer(query, red, answer, short, citations, False, faithfulness=rep.faithfulness,
                      retrieval_confidence=conf, generator=draft.model, latency_ms=_ms(t0), pii_found=pii,
                      events=events, unsupported=unsupported)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _cites(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", s)]


def _renumber(s: str, remap: dict[int, int]) -> str:
    return re.sub(r"\[(\d+)\]", lambda m: f"[{remap.get(int(m.group(1)), m.group(1))}]", s)


def _numbers_near(text: str, q: set[str], window: int = 8) -> set[str]:
    words = text.split()
    out = set()
    for i, w in enumerate(words):
        if set(content_tokens(w)) & q:
            out |= numbers(" ".join(words[max(0, i - window) : i + window]))
    return out
