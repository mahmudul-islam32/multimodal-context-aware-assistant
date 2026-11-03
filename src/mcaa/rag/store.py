"""Verified document store with approval records and versioning (F-05)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .guardrails import InjectionDetector

REQUIRED_META = ("doc_id", "title", "version", "approved_by", "approved_on", "source")


@dataclass
class Document:
    doc_id: str
    title: str
    version: str
    approved_by: str
    approved_on: str
    source: str
    body: str
    path: str = ""

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.body.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    version: str
    section: str
    text: str
    retired: bool = False

    def citation(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "title": self.title, "version": self.version, "section": self.section}


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    quarantined: list[tuple[str, str]] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)


def _vkey(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"[.\-]", str(v)))


def parse_markdown(path: Path) -> Document:
    raw = Path(path).read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        raise ValueError("missing front matter with approval record")
    meta = yaml.safe_load(m.group(1)) or {}
    missing = [k for k in REQUIRED_META if not meta.get(k)]
    if missing:
        raise ValueError(f"missing approval metadata: {', '.join(missing)}")
    date.fromisoformat(str(meta["approved_on"]))
    return Document(body=m.group(2), path=str(path), **{k: str(meta[k]) for k in REQUIRED_META})


def chunk_document(doc: Document, max_words: int = 160) -> list[Chunk]:
    """Split by markdown headings; long sections are split on paragraphs."""
    chunks: list[Chunk] = []
    heading = "Introduction"
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip()
        if not text:
            return
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        cur: list[str] = []
        for p in paras:
            if cur and len(" ".join(cur + [p]).split()) > max_words:
                chunks.append(_mk(doc, heading, "\n\n".join(cur), len(chunks)))
                cur = []
            cur.append(p)
        if cur:
            chunks.append(_mk(doc, heading, "\n\n".join(cur), len(chunks)))

    for line in doc.body.splitlines():
        h = re.match(r"^#{1,4}\s+(.*)", line)
        if h:
            flush()
            buf = []
            heading = h.group(1).strip()
        else:
            buf.append(line)
    flush()
    return chunks


def _mk(doc: Document, heading: str, text: str, i: int) -> Chunk:
    sec = heading if heading.lower().startswith("section") else f"section {heading}"
    return Chunk(f"{doc.doc_id}@{doc.version}#{i}", doc.doc_id, doc.title, doc.version, sec, text.strip())


class DocumentStore:
    def __init__(self, detector: InjectionDetector | None = None):
        self.detector = detector or InjectionDetector()
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []

    @property
    def active_chunks(self) -> list[Chunk]:
        return [c for c in self.chunks if not c.retired]

    def add(self, doc: Document, report: IngestReport | None = None) -> IngestReport:
        report = report or IngestReport()
        hit = self.detector.scan(doc.body)
        if hit.flagged:
            report.quarantined.append((doc.doc_id, "; ".join(hit.matches)))
            return report
        prev = self.documents.get(doc.doc_id)
        if prev and _vkey(doc.version) <= _vkey(prev.version):
            report.rejected.append((doc.doc_id, f"version {doc.version} not newer than {prev.version}"))
            return report
        if prev:
            for c in self.chunks:
                if c.doc_id == doc.doc_id and not c.retired:
                    c.retired = True
            report.retired.append(f"{prev.doc_id}@{prev.version}")
        self.documents[doc.doc_id] = doc
        self.chunks.extend(chunk_document(doc))
        report.indexed.append(f"{doc.doc_id}@{doc.version}")
        return report

    def ingest_dir(self, path: Path) -> IngestReport:
        report = IngestReport()
        files = sorted(Path(path).glob("**/*.md"))
        docs = []
        for f in files:
            try:
                docs.append(parse_markdown(f))
            except Exception as e:  # noqa: BLE001
                report.rejected.append((f.name, str(e)))
        for d in sorted(docs, key=lambda d: (d.doc_id, _vkey(d.version))):
            self.add(d, report)
        return report
