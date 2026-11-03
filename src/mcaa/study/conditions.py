"""Counterbalanced within-subjects schedule (condition order x task-set order)."""

from __future__ import annotations

import re

CONDITIONS = {
    "A": "context-aware (sensor + voice + adaptive output)",
    "B": "plain chatbot (same documents, retrieval and guardrails; text UI; no adaptation)",
}

# Balanced Latin square over (condition order) x (task-set pairing)
SEQUENCES = [
    [("A", "A"), ("B", "B")],
    [("B", "A"), ("A", "B")],
    [("A", "B"), ("B", "A")],
    [("B", "B"), ("A", "A")],
]

_PID = re.compile(r"^P\d{2,3}$")


def validate_pid(pid: str) -> str:
    """Participant IDs are pseudonyms like P07; names are never accepted."""
    if not _PID.match(pid):
        raise ValueError("participant id must look like P01 (pseudonym, no personal data)")
    return pid


def schedule_for(pid: str) -> list[dict]:
    n = int(validate_pid(pid)[1:])
    seq = SEQUENCES[(n - 1) % len(SEQUENCES)]
    return [{"block": i + 1, "condition": c, "task_set": t} for i, (c, t) in enumerate(seq)]
