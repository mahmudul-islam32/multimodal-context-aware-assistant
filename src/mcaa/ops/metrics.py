"""Minimal in-process metrics registry with Prometheus text exposition.

No dependency on prometheus_client: counters and summaries are kept in
memory and rendered by ``render()`` for ``GET /metrics``.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque

import numpy as np

HELP = {
    "mcaa_answers_total": "Answers produced, by context state and outcome",
    "mcaa_guardrail_events_total": "Guardrail events (injection blocked, PII redacted, output blocked, ...)",
    "mcaa_context_changes_total": "Context state transitions",
    "mcaa_feedback_total": "User feedback on answers",
    "mcaa_documents_total": "Document governance actions",
    "mcaa_answer_latency_ms": "Answer pipeline latency in milliseconds",
    "mcaa_har_inference_ms": "Activity model inference time per window in milliseconds",
}


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{str(v).replace(chr(34), "")}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class Metrics:
    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self.counters: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        self.summaries: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.summary_counts: dict[str, int] = defaultdict(int)
        self.summary_sums: dict[str, float] = defaultdict(float)

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        with self._lock:
            self.counters[name][tuple(sorted(labels.items()))] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self.summaries[name].append(value)
            self.summary_counts[name] += 1
            self.summary_sums[name] += value

    def quantile(self, name: str, q: float) -> float:
        vals = self.summaries.get(name)
        return float(np.quantile(list(vals), q)) if vals else 0.0

    def render(self) -> str:
        out: list[str] = []
        with self._lock:
            for name, series in sorted(self.counters.items()):
                out += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} counter"]
                for labels, v in sorted(series.items()):
                    out.append(f"{name}{_labels(dict(labels))} {v:g}")
            for name, vals in sorted(self.summaries.items()):
                out += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} summary"]
                arr = np.array(vals) if vals else np.array([0.0])
                for q in (0.5, 0.9, 0.99):
                    out.append(f'{name}{{quantile="{q}"}} {float(np.quantile(arr, q)):.3f}')
                out.append(f"{name}_sum {self.summary_sums[name]:.3f}")
                out.append(f"{name}_count {self.summary_counts[name]}")
        return "\n".join(out) + "\n"
