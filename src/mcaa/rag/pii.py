"""PII redaction before any external LLM call or log write (F-11)."""

from __future__ import annotations

import re

PATTERNS: list[tuple[str, str]] = [
    ("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    ("IBAN", r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b"),
    ("PHONE", r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,5}\)?[\s-]?)\d{3,4}[\s-]?\d{3,5}(?!\w)"),
    ("DATE", r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
    ("ID", r"\b(?:MRN|patient id|patient number|employee id|ssn|insurance (?:no|number))[:#\s]*[A-Z0-9-]{4,}\b"),
    ("NAME", r"\b(?i:(?:Mr|Mrs|Ms|Dr|Herr|Frau)\.?|patient|resident|my name is|I am|colleague)\s+([A-Z][a-zäöüß]+(?:\s+[A-Z][a-zäöüß]+)?)"),
]
_COMPILED = [(k, re.compile(p, re.I if k == "ID" else 0)) for k, p in PATTERNS]


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, list of entity types found)."""
    found = []
    for kind, rx in _COMPILED:
        if kind == "NAME":
            def repl(m):
                return m.group(0).replace(m.group(1), "[NAME]")
            new = rx.sub(repl, text)
        else:
            new = rx.sub(f"[{kind}]", text)
        if new != text:
            found.append(kind)
        text = new
    return text, found
