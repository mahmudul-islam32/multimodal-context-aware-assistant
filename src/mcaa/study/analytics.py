"""Aggregate the event log for the analytics dashboard (GET /api/analytics)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

GUARDRAIL_TYPES = ("injection_blocked", "pii_redacted", "output_blocked", "ungrounded_removed", "doc_quarantined")


def _payloads(df: pd.DataFrame) -> list[dict]:
    return [json.loads(p) for p in df["payload"]]


def summarize(ev: pd.DataFrame, recent: int = 12) -> dict:
    if ev.empty:
        ev = pd.DataFrame(columns=["id", "ts", "session_id", "participant", "condition", "task_set", "type", "payload"])
    answers = ev[ev.type == "answer"]
    ap = _payloads(answers)
    fb = _payloads(ev[ev.type == "feedback"])
    lat = np.array([a.get("latency_ms", 0.0) for a in ap]) if ap else np.array([])
    refused = [a for a in ap if a.get("refused")]
    faith = [a["faithfulness"] for a in ap if not a.get("refused")]
    helpful = [f["helpful"] for f in fb]
    fb_by_id = {f["answer_id"]: f["helpful"] for f in fb}

    states = Counter(a.get("context_state", "?") for a in ap)
    modality = Counter(a.get("modality", "?") for a in ap)
    reasons = Counter(a.get("reason") or "?" for a in refused)
    guard = {t: int((ev.type == t).sum()) for t in GUARDRAIL_TYPES}
    cites = Counter(c.split("#")[0] for a in ap for c in a.get("citations", []))

    changes = ev[ev.type.isin(["context_change", "context_override"])]
    timeline = []
    for (_, row), p in zip(changes.iterrows(), _payloads(changes), strict=True):
        timeline.append({"ts": float(row.ts), "state": p.get("to") or p.get("state") or "AUTO",
                         "override": row.type == "context_override"})

    per_min: Counter = Counter()
    for ts in answers.ts:
        per_min[int(ts // 60) * 60] += 1

    rec = []
    for (_, row), a in list(zip(answers.iterrows(), ap, strict=True))[-recent:][::-1]:
        rec.append({"ts": float(row.ts), "query": a.get("query", ""), "state": a.get("context_state"),
                    "modality": a.get("modality"), "refused": a.get("refused"), "reason": a.get("reason"),
                    "latency_ms": a.get("latency_ms"), "followup": a.get("interpreted_as") is not None,
                    "helpful": fb_by_id.get(a.get("answer_id"))})
    n = len(ap)
    return {
        "kpis": {
            "answers": n,
            "refusal_rate": len(refused) / n if n else 0.0,
            "blocked": sum(1 for a in ap if a.get("blocked")),
            "faithfulness": float(np.mean(faith)) if faith else None,
            "latency_ms_p50": float(np.percentile(lat, 50)) if len(lat) else None,
            "latency_ms_p95": float(np.percentile(lat, 95)) if len(lat) else None,
            "helpful_rate": float(np.mean(helpful)) if helpful else None,
            "feedback": len(helpful),
            "followups": sum(1 for a in ap if a.get("interpreted_as")),
            "briefings": int((ev.type == "briefing").sum()),
        },
        "by_state": dict(states),
        "by_modality": dict(modality),
        "refusal_reasons": dict(reasons.most_common(6)),
        "guardrails": guard,
        "top_sources": dict(cites.most_common(6)),
        "context_timeline": timeline[-60:],
        "answers_per_minute": [{"t": t, "n": c} for t, c in sorted(per_min.items())][-60:],
        "recent": rec,
    }
