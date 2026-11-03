"""Dialogue orchestrator: wires sensors -> HAR -> context, and
query -> guardrails -> RAG -> grounding -> policy, logging every decision."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Scenario, load_context_config, load_policy_config, load_scenario
from .context import ContextEngine
from .ops.metrics import Metrics
from .policy import OutputPolicy
from .rag import DocumentStore, RAGPipeline
from .rag.dialogue import ConversationMemory
from .rag.generator import get_generator
from .rag.governance import DocumentRegistry
from .rag.pii import redact
from .sensors import StreamBuffer
from .speech.tts import is_barge_in_command
from .study import EventLogger, schedule_for
from .vision import CameraContext

log = logging.getLogger("mcaa")

SAFETY_WORDS = re.compile(r"\b(lock|lockout|isolate|never|do not|don't|must|wear|gloves|glasses|verify|before|clean your hands|two carers|immediately|stop)\b", re.I)


@dataclass
class TaskProgress:
    task_set: str = "A"
    step: int = 0
    started_at: float | None = None
    done: bool = False
    errors: list[str] = field(default_factory=list)


class Assistant:
    def __init__(self, scenario: str | Scenario = "technician", har_model=None, logger: EventLogger | None = None, generator=None,
                 clock=time.time, runtime_dir: str | Path | None = None, metrics: Metrics | None = None):
        self.scenario = scenario if isinstance(scenario, Scenario) else load_scenario(scenario)
        self.clock = clock
        self.metrics = metrics or Metrics()
        self.store = DocumentStore()
        self.ingest_report = self.store.ingest_dir(self.scenario.docs_path())
        rt = Path(runtime_dir or os.environ.get("MCAA_RUNTIME_DIR", "data/runtime")) / "docs" / self.scenario.id
        self.registry = DocumentRegistry(self.store, rt)
        extra = self.registry.ingest_approved()
        self.ingest_report.indexed += extra.indexed
        self.ingest_report.retired += extra.retired
        self.memory = ConversationMemory()
        self.answers: dict[str, dict] = {}
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
        self.memory.clear()
        self.logger.start_session(participant, condition, task_set)
        self.logger.log("task_start", task=self.task.id, schedule=schedule_for(participant) if participant else None)
        self.context.set_step(self.current_step.manual)
        snap = self.snapshot()
        snap["briefing"] = self.briefing()
        return snap

    def set_step(self, index: int) -> dict:
        steps = self.task.steps
        prev = self.progress.step
        self.progress.step = max(0, min(index, len(steps) - 1))
        self.logger.log("step", step=self.current_step.id)
        self.touch(log_it=False)
        self.context.set_step(self.current_step.manual)
        snap = self.snapshot()
        if self.progress.step != prev:
            snap["briefing"] = self.briefing()
        return snap

    def briefing(self) -> dict | None:
        """Proactive safety briefing for the current step (condition A only).

        Retrieves the approved text for the step and prefers a sentence that
        carries a safety instruction. It goes through the same refusal gate
        and grounding check as any answer, and the output policy decides
        whether it is spoken.
        """
        if self.plain or not self.scenario.proactive_briefings:
            return None
        step = self.current_step
        ans = self.rag.ask(step.text, step_hint=step.doc_hint or "")
        if ans.refused or not ans.citations:
            return None
        sents = re.findall(r"[^.!?]+[.!?]\s*(?:\[\d+\])?", ans.answer)
        safety = [s.strip() for s in sents if SAFETY_WORDS.search(s)]
        text = safety[0] if safety else ans.short_answer
        ctx = self.context.evaluate()
        plan = self.policy.plan(ctx.state, text, text, ans.citations, False)
        self.logger.log("briefing", step=step.id, text=text, state=ctx.state, spoken=plan.speak)
        return {"step": step.id, "text": text, "plan": plan.to_dict()}

    def feedback(self, answer_id: str, helpful: bool, comment: str = "") -> dict:
        if answer_id not in self.answers:
            raise KeyError(answer_id)
        info = self.answers[answer_id]
        self.logger.log("feedback", answer_id=answer_id, helpful=helpful, text=comment[:500],
                        refused=info["refused"], state=info["state"])
        self.metrics.inc("mcaa_feedback_total", helpful=str(helpful).lower())
        return {"ok": True}

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
        self.metrics.observe("mcaa_har_inference_ms", dt)
        prev = self.context.current.state
        st = self.context.update_activity(self.last_activity)
        if st.state != prev:
            self.logger.log("context_change", frm=prev, to=st.state, reason=st.reason, confidence=st.confidence)
            self.metrics.inc("mcaa_context_changes_total", to=st.state)
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
        res = self.memory.resolve(text)
        ans = self.rag.ask(res.retrieval_query, step_hint=step.doc_hint or "", focus=text if res.expanded else None)
        if res.expanded:  # log and show the user's own words, not the weighted retrieval string
            ans.query = text
            ans.redacted_query = redact(text)[0]
        self.memory.add(text, res.display, ans.refused)
        answer_id = uuid.uuid4().hex[:12]
        interpreted = res.display if res.expanded else None
        self.answers[answer_id] = {"refused": ans.refused, "state": ctx.state}
        plan = self.policy.plan(ctx.state, ans.answer, ans.short_answer, ans.citations, ans.refused, plain=self.plain)
        for ev in ans.events:
            self.logger.log(ev["type"], **{k: v for k, v in ev.items() if k != "type"})
            if ev["type"] in ("injection_blocked", "pii_redacted", "output_blocked", "ungrounded_removed"):
                self.metrics.inc("mcaa_guardrail_events_total", type=ev["type"])
        outcome = "blocked" if ans.blocked else "refused" if ans.refused else "answered"
        self.metrics.inc("mcaa_answers_total", state="PLAIN" if self.plain else ctx.state, outcome=outcome)
        self.metrics.observe("mcaa_answer_latency_ms", ans.latency_ms)
        self.logger.log(
            "answer",
            answer_id=answer_id,
            interpreted_as=interpreted,
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
        return {"answer_id": answer_id, "interpreted_as": interpreted, "answer": ans.to_dict(), "plan": plan.to_dict(),
                "context": ctx.to_dict(), "condition": self.condition}

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
