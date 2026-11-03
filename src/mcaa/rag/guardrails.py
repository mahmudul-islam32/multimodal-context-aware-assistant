"""Prompt-injection detection and output checks (F-08)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS: list[tuple[str, float]] = [
    (r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(previous|prior|above|earlier|all|your|the|system)\b[^.]{0,20}\b(instructions?|rules?|prompts?|guidelines?|polic(y|ies))", 1.0),
    (r"\byou are now\b|\bfrom now on,? you\b|\bact as (an? )?(unrestricted|different|new)\b", 0.8),
    (r"\b(system|developer|hidden) (prompt|message|instructions?)\b", 0.7),
    (r"\b(reveal|print|show|repeat|leak)\b[^.]{0,30}\b(prompt|instructions|configuration|api key|password|secret)", 0.9),
    (r"\b(jailbreak|developer mode|dan mode|do anything now)\b", 1.0),
    (r"\bpretend (that |to be )?\b", 0.5),
    (r"\bnew instructions?\b|\binstead,? (say|answer|respond|tell)\b", 0.7),
    (r"\b(when|if) (the )?(assistant|ai|model|llm) (is asked|reads this)\b", 1.0),
    (r"\b(assistant|ai|model) (must|should) (now )?(say|answer|reply|tell|recommend)\b", 0.9),
    (r"<\s*/?\s*(script|system|instructions?)\b|\[/?(inst|system)\]|###\s*(system|instruction)", 0.9),
    (r"\b(bypass|disable|turn off|skip)\b[^.]{0,25}\b(safety|guardrails?|filters?|lockout|interlock|alarms?)\b", 0.6),
    (r"\b(no need to|don'?t) (cite|use) (the )?(sources|documents)\b", 0.7),
]


@dataclass
class ScanResult:
    score: float
    flagged: bool
    matches: list[str] = field(default_factory=list)


class InjectionDetector:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.patterns = [(re.compile(p, re.I), w) for p, w in INJECTION_PATTERNS]

    def scan(self, text: str) -> ScanResult:
        score, hits = 0.0, []
        for rx, w in self.patterns:
            m = rx.search(text)
            if m:
                score += w
                hits.append(m.group(0)[:60])
        return ScanResult(round(score, 3), score >= self.threshold, hits)


UNSAFE_OUTPUT = [
    r"\b(bypass|defeat|jumper|bridge)\b[^.]{0,30}\b(interlock|safety|lockout|guard|alarm)",
    r"\b(without|skip(ping)?|no need for)\b[^.]{0,20}\b(lockout|isolation|ppe|gloves|hand hygiene|verification)",
    r"\b(double|increase) the (dose|dosage)\b",
]


@dataclass
class OutputCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)


class OutputChecker:
    """Block unsafe instructions that are not literally present in the sources,
    and any leakage of the system prompt."""

    def __init__(self, system_prompt_marker: str = "MCAA-SYSTEM"):
        self.unsafe = [re.compile(p, re.I) for p in UNSAFE_OUTPUT]
        self.marker = system_prompt_marker

    def check(self, answer: str, sources: list[str]) -> OutputCheck:
        reasons = []
        src = " ".join(sources).lower()
        for rx in self.unsafe:
            for m in rx.finditer(answer):
                if m.group(0).lower() not in src:
                    reasons.append(f"unsafe instruction: '{m.group(0)}'")
        if self.marker in answer:
            reasons.append("system prompt leakage")
        return OutputCheck(not reasons, reasons)
