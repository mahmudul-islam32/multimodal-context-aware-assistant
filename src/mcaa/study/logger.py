"""SQLite event log for audit and study analysis, with export and deletion."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

from ..rag.pii import redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  session_id TEXT,
  participant TEXT,
  condition TEXT,
  task_set TEXT,
  type TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_participant ON events(participant);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(type);
"""

# event types that may contain free text from the user -> redacted before storage
_TEXT_FIELDS = ("query", "text", "transcript")


class EventLogger:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.session: dict = {}

    def start_session(self, participant: str | None, condition: str, task_set: str) -> str:
        sid = f"{participant or 'anon'}-{condition}{task_set}-{int(time.time())}"
        self.session = {"session_id": sid, "participant": participant, "condition": condition, "task_set": task_set}
        return sid

    def end_session(self) -> None:
        self.session = {}

    def log(self, type_: str, **payload) -> None:
        for f in _TEXT_FIELDS:
            if isinstance(payload.get(f), str):
                payload[f], _ = redact(payload[f])
        row = (
            time.time(),
            self.session.get("session_id"),
            self.session.get("participant"),
            self.session.get("condition"),
            self.session.get("task_set"),
            type_,
            json.dumps(payload, default=str),
        )
        with self._lock:
            self.db.execute(
                "INSERT INTO events (ts, session_id, participant, condition, task_set, type, payload) VALUES (?,?,?,?,?,?,?)",
                row,
            )
            self.db.commit()

    def events(self, type_: str | None = None, participant: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM events WHERE 1=1", []
        if type_:
            q += " AND type = ?"
            args.append(type_)
        if participant:
            q += " AND participant = ?"
            args.append(participant)
        with self._lock:
            df = pd.read_sql_query(q + " ORDER BY id", self.db, params=args)
        return df

    def export(self, out_dir: str | Path, fmt: str = "csv") -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ev = self.events()
        tasks = task_summary(ev)
        paths = []
        for name, df in (("events", ev), ("tasks", tasks)):
            p = out / f"{name}.{ 'parquet' if fmt == 'parquet' else 'csv'}"
            if fmt == "parquet":
                df.to_parquet(p, index=False)
            else:
                df.to_csv(p, index=False)
            paths.append(p)
        return paths

    def delete_participant(self, participant: str) -> int:
        with self._lock:
            cur = self.db.execute("DELETE FROM events WHERE participant = ?", (participant,))
            self.db.commit()
        return cur.rowcount

    def purge_older_than(self, days: float) -> int:
        with self._lock:
            cur = self.db.execute("DELETE FROM events WHERE ts < ?", (time.time() - days * 86400,))
            self.db.commit()
        return cur.rowcount


def task_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per session: task time, errors, queries, refusals, overrides."""
    if ev.empty:
        return pd.DataFrame(columns=["session_id", "participant", "condition", "task_set", "task_time_s", "errors", "queries", "refusals", "overrides"])
    rows = []
    for sid, g in ev.dropna(subset=["session_id"]).groupby("session_id"):
        start = g.loc[g.type == "task_start", "ts"]
        end = g.loc[g.type == "task_end", "ts"]
        rows.append({
            "session_id": sid,
            "participant": g.participant.iloc[0],
            "condition": g.condition.iloc[0],
            "task_set": g.task_set.iloc[0],
            "task_time_s": float(end.max() - start.min()) if len(start) and len(end) else None,
            "errors": int((g.type == "error").sum()),
            "queries": int((g.type == "answer").sum()),
            "refusals": int(sum(json.loads(p).get("refused", False) for p in g.loc[g.type == "answer", "payload"])),
            "overrides": int((g.type == "context_override").sum()),
        })
    return pd.DataFrame(rows)
