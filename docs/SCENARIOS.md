# Scenario guide

A scenario is a YAML file in `configs/scenarios/` plus a folder of approved documents. It is validated at start-up (`mcaa.config.Scenario`), and invalid packs fail fast.

```yaml
id: technician
name: Field technician - pump seal replacement
simulator_only: false          # true requires a `warning`, shown as a permanent banner
documents_dir: data/docs/technician
escalation: {name: the shift supervisor, contact: ext. 4410}
context_config: context.yaml   # optional, relative to configs/
policy_config: policy.yaml
task_sets:                     # two comparable sets for counterbalancing (A, B)
  A:
    id: seal
    title: Replace the mechanical seal on pump P-101
    steps:
      - {id: s1, text: "Lock out the pump", manual: true, doc_hint: lockout isolate valves}
```

- `manual: true` marks steps that need the hands. With no touch for 20 s, the context becomes `HANDS_BUSY`.
- `doc_hint` adds retrieval keywords for very short questions ("how?").

## Documents

Markdown with front matter. Files without a complete approval record are rejected.

```markdown
---
doc_id: pump-p101-manual
title: Pump P-101 maintenance manual
version: "2.1"
approved_by: K. Weber, maintenance lead
approved_on: 2026-03-01
source: where it came from
---
# Section 4.3 Seal replacement
...
```

- Headings become citation sections ("Section 4.3 Seal replacement"). Use one heading per procedure.
- To publish a new version, add the file with a higher `version`. The old chunks are retired automatically. An older or equal version is rejected.
- Documents that contain instruction-like text ("ignore previous instructions", "the assistant must say …") are quarantined, and the event is logged.

Run `mcaa-eval` after changes. Add gold questions for your scenario to `eval/gold_questions.yaml`.
