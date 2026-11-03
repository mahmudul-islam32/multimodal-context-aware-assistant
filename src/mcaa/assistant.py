"""Dialogue orchestrator: wires sensors -> HAR -> context, and
query -> guardrails -> RAG -> grounding -> policy, logging every decision."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Scenario, load_context_config, load_policy_config, load_scenario
from .context import ContextEngine
from .policy import OutputPolicy
from .rag import DocumentStore, RAGPipeline
from .rag.generator import get_generator
from .sensors import StreamBuffer
from .speech.tts import is_barge_in_command
from .study import EventLogger, schedule_for
from .vision import CameraContext

log = logging.getLogger("mcaa")


@dataclass
class TaskProgress:
    task_set: str = "A"
    step: int = 0
    started_at: float | None = None
    done: bool = False
    errors: list[str] = field(default_factory=list)


class Assistant:
    def __init__(self, scenario: str | Scenario = "technician", har_model=None, logger: EventLogger | None = None, generator=None, clock=time.time):
        self.scenario = scenario if isinstance(scenario, Scenario) else load_scenario(scenario)
        self.clock = clock
        self.store = DocumentStore()
        self.ingest_report = self.store.ingest_dir(self.scenario.docs_path())
        self.rag = RAGPipeline(self.store, self.scenario.escalation, generator or get_generator())
        self.context = ContextEngine(load_context_config(self.scenario.context_config), clock=clock)
        self.policy = OutputPolicy(load_policy_config(self.scenario.policy_config))
        self.logger = logger or EventLogger(os.environ.get("MCAA_DB", ":memory:"))
        self.har = har_model
        self.sensors = StreamBuffer()
        self.camera = CameraContext()
        self.condition = "A"
        self.participant: str | None = None
        self.progress = TaskProgress()
        self.last_activity: dict[str, float] = {}
        self._log_ingest()

    # ---------------------------------------------------------------- setup
    def _log_ingest(self):
        r = self.ingest_report
        self.logger.log("ingest", indexed=r.indexed, rejected=r.rejected, quarantined=r.quarantined, retired=r.retired)
        for doc, why in r.quarantined:
            self.logger.log("injection_blocked", where="document", doc_id=doc, matches=why)

    def ensure_har(self):
        if self.har is None:
            path = Path(os.environ.get("MCAA_HAR_MODEL", "models/har_rf.pkl"))
            if path.exists():
                from .har.models import FeatureModel

                self.har = FeatureModel.load(path)
            else:
                from .har.train import train_default_model

                log.info("training default HAR model on synthetic data (no %s found)", path)
                self.har = train_default_model()
        return self.har

    @property
    def plain(self) -> bool:
        return self.condition == "B"

    @property
    def task(self):
        return self.scenario.task_sets[self.progress.task_set]

    @property
    def current_step(self):
        steps = self.task.steps
        return steps[min(self.progress.step, len(steps) - 1)]

    # ---------------------------------------------------------------- study
    def start_task(self, task_set: str = "A", condition: str = "A", participant: str | None = None) -> dict:
        self.condition = condition
        self.participant = participant
        self.progress = TaskProgress(task_set=task_set, started_at=self.clock())
        self.logger.start_session(participant, condition, task_set)
        self.logger.log("task_start", task=self.task.id, schedule=schedule_for(participant) if participant else None)
        self.context.set_step(self.current_step.manual)
        return self.snapshot()

    def set_step(self, index: int) -> dict:
        steps = self.task.steps
        self.progress.step = max(0, min(index, len(steps) - 1))
        self.logger.log("step", step=self.current_step.id)
        self.touch(log_it=False)
        self.context.set_step(self.current_step.manual)
        return self.snapshot()

    def complete_task(self) -> dict:
        self.progress.done = True
        self.logger.log("task_end", task=self.task.id, duration_s=self.clock() - (self.progress.started_at or self.clock()))
        self.logger.end_session()
        return self.snapshot()

    def record_error(self, kind: str, note: str = "") -> None:
        self.progress.errors.append(kind)
        self.logger.log("error", kind=kind, note=note, step=self.current_step.id)

    # ---------------------------------------------------------------- inputs
    def touch(self, log_it: bool = True) -> None:
        self.context.touch()
        if log_it:
            self.logger.log("touch")

    def sensor_sample(self, sample: dict) -> dict | None:
        """Feed one IMU sample; returns a context update when a window completes."""
        w = self.sensors.push(sample)
        if w is None:
            return None
        return self.classify_window(w)

    def classify_window(self, window: np.ndarray) -> dict:
        model = self.ensure_har()
        t0 = time.perf_counter()
        proba = model.predict_proba(window[None])[0]
        dt = (time.perf_counter() - t0) * 1000
        self.last_activity = {str(c): float(p) for c, p in zip(model.classes_, proba, strict=True)}
        prev = self.context.current.state
        st = self.context.update_activity(self.last_activity)
        if st.state != prev:
            self.logger.log("context_change", frm=prev, to=st.state, reason=st.reason, confidence=st.confidence)
        return {"activity": self.last_activity, "har_ms": round(dt, 2), "context": st.to_dict()}

    def camera_frame(self, frame: np.ndarray) -> dict:
        cues = self.camera.process(frame)
        self.context.update_camera(cues.hands_busy)
        return {"hands_in_view": cues.hands_in_view, "motion": cues.motion, "hands_busy": cues.hands_busy}

    def override(self, state: str | None, duration_s: float | None = None) -> dict:
        st = self.context.override(state, duration_s)
        self.logger.log("context_override", state=state, duration_s=duration_s)
        return st.to_dict()

    # ---------------------------------------------------------------- ask
    def ask(self, text: str, source: str = "text", asr_confidence: float | None = None, noise_dbfs: float | None = None) -> dict:
        if source == "voice" and is_barge_in_command(text):
            self.logger.log("barge_in", source="voice")
            return {"barge_in": True}
        if source == "text":
            self.touch(log_it=False)
        if asr_confidence is not None or noise_dbfs is not None:
            self.context.update_audio(noise_dbfs, asr_confidence)
        ctx = self.context.evaluate()
        step = self.current_step
        self.logger.log("query", query=text, source=source, state=ctx.state, step=step.id)
        ans = self.rag.ask(text, step_hint=step.doc_hint or "")
        plan = self.policy.plan(ctx.state, ans.answer, ans.short_answer, ans.citations, ans.refused, plain=self.plain)
        for ev in ans.events:
            self.logger.log(ev["type"], **{k: v for k, v in ev.items() if k != "type"})
        self.logger.log(
            "answer",
            query=ans.redacted_query,
            refused=ans.refused,
            reason=ans.refusal_reason,
            blocked=ans.blocked,
            faithfulness=ans.faithfulness,
            citations=[c["chunk_id"] for c in ans.citations],
            generator=ans.generator,
            context_state=ctx.state,
            context_confidence=ctx.confidence,
            policy_version=plan.policy_version,
            modality=plan.modality,
            voice_words=len(plan.voice_text.split()),
            latency_ms=ans.latency_ms,
        )
        return {"answer": ans.to_dict(), "plan": plan.to_dict(), "context": ctx.to_dict(), "condition": self.condition}

    # ---------------------------------------------------------------- state
    def snapshot(self) -> dict:
        sc = self.scenario
        return {
            "scenario": {"id": sc.id, "name": sc.name, "simulator_only": sc.simulator_only, "warning": sc.warning,
                         "escalation": sc.escalation.model_dump()},
            "condition": self.condition,
            "participant": self.participant,
            "task": {"set": self.progress.task_set, "title": self.task.title, "step": self.progress.step,
                     "steps": [s.model_dump() for s in self.task.steps], "done": self.progress.done},
            "context": self.context.current.to_dict(),
            "activity": self.last_activity,
            "sensors": self.sensors.stats.summary(),
            "camera": {"enabled": self.camera.enabled},
            "documents": {"indexed": self.ingest_report.indexed, "rejected": self.ingest_report.rejected,
                          "quarantined": self.ingest_report.quarantined, "retired": self.ingest_report.retired},
            "policy_version": self.policy.cfg.version,
        }
