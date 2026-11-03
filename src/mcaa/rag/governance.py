"""Document governance: upload -> scan -> review/diff -> approve (four-eyes) -> index.

Uploaded documents never reach the index directly. They are stored as
*pending* under ``data/runtime/docs/<scenario>/pending``; a reviewer who is
not the uploader approves them, which stamps the approval record into the
front matter, writes the file to ``approved/`` (so approvals survive
restarts) and re-indexes. Injection markers quarantine the upload at once.
Every action is returned as an audit event for the event log.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from .store import Document, DocumentStore, _vkey, parse_markdown

_FM = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


class GovernanceError(ValueError):
    pass


@dataclass
class PendingDoc:
    doc_id: str
    title: str
    version: str
    source: str
    uploaded_by: str
    uploaded_at: str
    status: str  # pending | quarantined | rejected | approved
    body: str
    findings: list[str] = field(default_factory=list)
    decided_by: str | None = None
    decided_at: str | None = None
    note: str = ""

    def public(self) -> dict:
        d = asdict(self)
        d["words"] = len(self.body.split())
        d.pop("body")
        return d


class DocumentRegistry:
    def __init__(self, store: DocumentStore, runtime_dir: Path):
        self.store = store
        self.dir = Path(runtime_dir)
        self.pending: dict[str, PendingDoc] = {}
        (self.dir / "approved").mkdir(parents=True, exist_ok=True)
        (self.dir / "pending").mkdir(parents=True, exist_ok=True)
        self._load_pending()

    # ------------------------------------------------------------ persistence
    def _load_pending(self) -> None:
        for f in sorted((self.dir / "pending").glob("*.yaml")):
            data = yaml.safe_load(f.read_text())
            self.pending[data["doc_id"]] = PendingDoc(**data)

    def _save_pending(self, p: PendingDoc) -> None:
        (self.dir / "pending" / f"{p.doc_id}.yaml").write_text(yaml.safe_dump(asdict(p), sort_keys=False, allow_unicode=True))

    def ingest_approved(self):
        """Re-index documents approved through this workflow in earlier runs."""
        return self.store.ingest_dir(self.dir / "approved")

    # ------------------------------------------------------------ actions
    def upload(self, raw: str, uploaded_by: str) -> tuple[PendingDoc, dict]:
        m = _FM.match(raw.replace("\r\n", "\n"))
        if not m:
            raise GovernanceError("document needs YAML front matter with doc_id, title, version, source")
        meta = yaml.safe_load(m.group(1)) or {}
        missing = [k for k in ("doc_id", "title", "version", "source") if not meta.get(k)]
        if missing:
            raise GovernanceError(f"missing front matter fields: {', '.join(missing)}")
        doc_id = str(meta["doc_id"]).strip().lower()
        if not _SAFE_ID.match(doc_id):
            raise GovernanceError("doc_id may contain only a-z, 0-9, '.', '_' and '-'")
        version = str(meta["version"])
        active = self.store.documents.get(doc_id)
        if active and _vkey(version) <= _vkey(active.version):
            raise GovernanceError(f"version {version} is not newer than the active version {active.version}")
        body = m.group(2)
        scan = self.store.detector.scan(body)
        p = PendingDoc(
            doc_id=doc_id, title=str(meta["title"]), version=version, source=str(meta["source"]),
            uploaded_by=uploaded_by, uploaded_at=_now(), status="quarantined" if scan.flagged else "pending",
            body=body, findings=scan.matches,
        )
        self.pending[doc_id] = p
        self._save_pending(p)
        ev = {"type": "doc_quarantined" if scan.flagged else "doc_uploaded", "doc_id": doc_id, "version": version,
              "by": uploaded_by, "findings": scan.matches}
        return p, ev

    def approve(self, doc_id: str, approver: str, note: str = "") -> tuple[Document, dict]:
        p = self._get(doc_id)
        if p.status != "pending":
            raise GovernanceError(f"cannot approve a document with status '{p.status}'")
        if approver.strip().lower() == p.uploaded_by.strip().lower():
            raise GovernanceError("four-eyes rule: the uploader cannot approve their own document")
        meta = {"doc_id": p.doc_id, "title": p.title, "version": p.version, "approved_by": approver,
                "approved_on": date.today().isoformat(), "source": p.source}
        path = self.dir / "approved" / f"{p.doc_id}_v{p.version}.md"
        path.write_text("---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n" + p.body)
        doc = parse_markdown(path)
        rep = self.store.add(doc)
        if not rep.indexed:
            path.unlink(missing_ok=True)
            raise GovernanceError(f"indexing failed: {rep.rejected or rep.quarantined}")
        p.status, p.decided_by, p.decided_at, p.note = "approved", approver, _now(), note
        self._save_pending(p)
        return doc, {"type": "doc_approved", "doc_id": doc_id, "version": p.version, "by": approver,
                     "retired": rep.retired, "note": note}

    def reject(self, doc_id: str, reviewer: str, note: str) -> dict:
        p = self._get(doc_id)
        if p.status not in ("pending", "quarantined"):
            raise GovernanceError(f"cannot reject a document with status '{p.status}'")
        p.status, p.decided_by, p.decided_at, p.note = "rejected", reviewer, _now(), note
        self._save_pending(p)
        return {"type": "doc_rejected", "doc_id": doc_id, "version": p.version, "by": reviewer, "note": note}

    def diff(self, doc_id: str) -> dict:
        p = self._get(doc_id)
        active = self.store.documents.get(doc_id)
        old = active.body if active and p.status != "approved" else ""
        lines = list(difflib.unified_diff(
            old.splitlines(), p.body.splitlines(),
            fromfile=f"{doc_id}@{active.version}" if active else "(new document)",
            tofile=f"{doc_id}@{p.version}", lineterm="", n=1,
        ))
        added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        return {"doc_id": doc_id, "from": active.version if active else None, "to": p.version,
                "added": added, "removed": removed, "diff": "\n".join(lines)}

    def listing(self) -> dict:
        active = []
        for d in sorted(self.store.documents.values(), key=lambda d: d.doc_id):
            n = sum(1 for c in self.store.active_chunks if c.doc_id == d.doc_id)
            active.append({"doc_id": d.doc_id, "title": d.title, "version": d.version, "approved_by": d.approved_by,
                           "approved_on": d.approved_on, "source": d.source, "chunks": n, "checksum": d.checksum})
        retired = sorted({f"{c.doc_id}@{c.version}" for c in self.store.chunks if c.retired})
        return {"active": active, "retired": retired,
                "queue": [p.public() for p in sorted(self.pending.values(), key=lambda p: p.uploaded_at, reverse=True)]}

    def _get(self, doc_id: str) -> PendingDoc:
        if doc_id not in self.pending:
            raise GovernanceError(f"no uploaded document '{doc_id}'")
        return self.pending[doc_id]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
