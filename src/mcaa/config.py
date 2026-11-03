"""F-13: scenario, context and policy configuration (validated on start)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__import__("os").environ.get("MCAA_CONFIG_DIR", ROOT / "configs"))

ContextStateName = Literal["IDLE", "HANDS_BUSY", "MOVING", "NOISY"]


class Step(BaseModel):
    id: str
    text: str
    manual: bool = False  # hands needed for this step
    doc_hint: str | None = None


class Task(BaseModel):
    id: str
    title: str
    steps: list[Step] = Field(min_length=1)


class Escalation(BaseModel):
    name: str
    contact: str


class Scenario(BaseModel):
    id: str
    name: str
    description: str = ""
    simulator_only: bool = False
    warning: str | None = None
    documents_dir: str
    escalation: Escalation
    task_sets: dict[str, Task] = Field(min_length=1)
    context_config: str = "context.yaml"
    policy_config: str = "policy.yaml"
    language: str = "en"
    proactive_briefings: bool = True  # speak the key safety line when a step starts

    @model_validator(mode="after")
    def _driver_warning(self):
        if self.simulator_only and not self.warning:
            raise ValueError("simulator_only scenarios must carry a visible warning")
        return self

    def docs_path(self) -> Path:
        p = Path(self.documents_dir)
        return p if p.is_absolute() else ROOT / p


class Condition(BaseModel):
    """Leaf of a context rule: signal comparisons, all must hold."""

    model_config = {"extra": "allow"}


class Rule(BaseModel):
    state: ContextStateName
    when: dict[str, Any] = Field(default_factory=dict)
    when_any: list[dict[str, Any]] = Field(default_factory=list)
    confidence_from: str | None = None
    reason: str = ""


class ContextConfig(BaseModel):
    version: str
    min_dwell_s: float = 4.0
    confirm_updates: int = 2
    ema_alpha: float = 0.5
    rules: list[Rule] = Field(min_length=1)

    @model_validator(mode="after")
    def _has_default(self):
        if self.rules[-1].when or self.rules[-1].when_any:
            raise ValueError("last context rule must be an unconditional default")
        return self


class StatePolicy(BaseModel):
    modality: Literal["voice_first", "screen", "screen_first", "both"]
    max_voice_words: int = 25
    voice: Literal["required", "optional", "off"] = "optional"
    screen_detail: Literal["full", "summary"] = "full"
    citations: Literal["full", "short"] = "full"
    visual_cue: bool = False
    defer_details: bool = False


class PolicyConfig(BaseModel):
    version: str
    states: dict[ContextStateName, StatePolicy]
    plain_chatbot: StatePolicy

    @model_validator(mode="after")
    def _all_states(self):
        missing = {"IDLE", "HANDS_BUSY", "MOVING", "NOISY"} - set(self.states)
        if missing:
            raise ValueError(f"policy missing states: {sorted(missing)}")
        return self


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() or p.exists() else CONFIG_DIR / name


def load_scenario(name_or_path: str) -> Scenario:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / "scenarios" / f"{name_or_path}.yaml"
    return Scenario.model_validate(_load_yaml(p))


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "scenarios").glob("*.yaml"))


def load_context_config(name: str = "context.yaml") -> ContextConfig:
    return ContextConfig.model_validate(_load_yaml(resolve(name)))


def load_policy_config(name: str = "policy.yaml") -> PolicyConfig:
    return PolicyConfig.model_validate(_load_yaml(resolve(name)))
