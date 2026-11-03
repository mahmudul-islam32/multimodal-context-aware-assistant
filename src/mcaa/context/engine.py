"""Fuse activity, UI, task step, noise and camera cues into one explicit state.

Rules are declarative (configs/context.yaml), evaluated top to bottom; the
first match wins. Hysteresis: a candidate must be proposed ``confirm_updates``
times in a row and at least ``min_dwell_s`` must have passed since the last
change. A user override always wins until it expires or is cleared.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import ContextConfig, load_context_config
from ..sensors.synthetic import ACTIVITY_TO_TASK


@dataclass
class ContextSignals:
    p_idle: float = 1.0
    p_moving: float = 0.0
    p_hands_busy: float = 0.0
    activity: str = "UNKNOWN"
    noise_dbfs: float = -60.0
    asr_confidence: float = 1.0
    last_touch_at: float | None = None
    step_manual: bool = False
    camera_hands_busy: float = 0.0
    now: float = 0.0

    @property
    def no_touch_s(self) -> float:
        return 1e9 if self.last_touch_at is None else self.now - self.last_touch_at

    def as_values(self) -> dict[str, Any]:
        d = asdict(self)
        d["no_touch_s"] = self.no_touch_s
        return d


@dataclass
class ContextState:
    state: str
    confidence: float
    reason: str
    since: float
    overridden: bool = False
    candidate: str | None = None
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _check(cond: dict[str, Any], values: dict[str, Any]) -> bool:
    for key, target in cond.items():
        if key.endswith("_gt"):
            if not values[key[:-3]] > target:
                return False
        elif key.endswith("_lt"):
            if not values[key[:-3]] < target:
                return False
        elif values[key] != target:
            return False
    return True


class ContextEngine:
    def __init__(self, config: ContextConfig | None = None, clock=time.time):
        self.cfg = config or load_context_config()
        self.clock = clock
        self.signals = ContextSignals(now=clock())
        self.current = ContextState("IDLE", 1.0, "start", since=self.clock())
        self._pending: tuple[str, int] | None = None
        self._override: tuple[str, float | None] | None = None
        self.history: list[ContextState] = [self.current]

    # ---- inputs ------------------------------------------------------
    def update_activity(self, proba: dict[str, float]) -> ContextState:
        """Activity class probabilities from the HAR model (per window)."""
        agg = {"idle": 0.0, "moving": 0.0, "hands_busy": 0.0}
        for cls, p in proba.items():
            agg[ACTIVITY_TO_TASK.get(cls, "idle")] += p
        a = self.cfg.ema_alpha
        s = self.signals
        s.p_idle = a * agg["idle"] + (1 - a) * s.p_idle
        s.p_moving = a * agg["moving"] + (1 - a) * s.p_moving
        s.p_hands_busy = a * agg["hands_busy"] + (1 - a) * s.p_hands_busy
        s.activity = max(proba, key=proba.get) if proba else "UNKNOWN"
        return self.evaluate()

    def update_audio(self, noise_dbfs: float | None = None, asr_confidence: float | None = None) -> ContextState:
        if noise_dbfs is not None:
            self.signals.noise_dbfs = noise_dbfs
        if asr_confidence is not None:
            self.signals.asr_confidence = asr_confidence
        return self.evaluate()

    def touch(self) -> ContextState:
        self.signals.last_touch_at = self.clock()
        return self.evaluate()

    def set_step(self, manual: bool) -> ContextState:
        self.signals.step_manual = manual
        return self.evaluate()

    def update_camera(self, hands_busy: float) -> ContextState:
        self.signals.camera_hands_busy = hands_busy
        return self.evaluate()

    def override(self, state: str | None, duration_s: float | None = None) -> ContextState:
        now = self.clock()
        if state is None:
            self._override = None
        else:
            self._override = (state, None if duration_s is None else now + duration_s)
            self._set(ContextState(state, 1.0, "manual override", now, overridden=True))
        return self.evaluate()

    # ---- core --------------------------------------------------------
    def propose(self) -> tuple[str, float, str]:
        values = self.signals.as_values()
        for rule in self.cfg.rules:
            hit = _check(rule.when, values) if rule.when else not rule.when_any
            if rule.when_any:
                hit = any(_check(c, values) for c in rule.when_any)
            if hit:
                conf = float(values[rule.confidence_from]) if rule.confidence_from else 0.8
                return rule.state, max(0.0, min(1.0, conf)), rule.reason or rule.state
        raise RuntimeError("unreachable: default rule required")

    def _set(self, st: ContextState) -> None:
        self.current = st
        self.history.append(st)

    def evaluate(self) -> ContextState:
        now = self.clock()
        self.signals.now = now
        if self._override:
            state, until = self._override
            if until is None or now < until:
                self.current.signals = self.signals.as_values()
                return self.current
            self._override = None
        cand, conf, reason = self.propose()
        cur = self.current
        if cand == cur.state:
            self._pending = None
            cur.confidence, cur.reason, cur.candidate = conf, reason, None
            cur.overridden = False
            cur.signals = self.signals.as_values()
            return cur
        n = self._pending[1] + 1 if self._pending and self._pending[0] == cand else 1
        self._pending = (cand, n)
        dwell_ok = (now - cur.since) >= self.cfg.min_dwell_s or cur.overridden
        if n >= self.cfg.confirm_updates and dwell_ok:
            self._pending = None
            self._set(ContextState(cand, conf, reason, now, signals=self.signals.as_values()))
        else:
            cur.candidate = cand
            cur.signals = self.signals.as_values()
        return self.current
