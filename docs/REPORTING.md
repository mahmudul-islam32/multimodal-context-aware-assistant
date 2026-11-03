# Reporting, Documentation and Reproducibility Plan

> Status: the technical documentation is in place for v0.1 (model card, scenario guide, safety case, changelog). The paper is still planned.

## Paper

Target: a short or full paper in a standard conference format, e.g. ACM `acmart` (CHI late-breaking work, MobileHCI, IUI) or IEEE conference template, written in LaTeX. Length: 4 to 8 pages plus references, depending on venue.

### Outline

1. **Abstract** (150 to 200 words)
2. **Introduction:** problem of information access with busy hands; safety risk of ungrounded LLM answers; contributions.
3. **Related work:** context-aware and multimodal interfaces; human activity recognition with wearables; RAG, hallucination and refusal; prompt-injection defences; trust in automation.
4. **System design:** scenario, architecture, context engine, output policy, safety and privacy design.
5. **Technical evaluation:** HAR, ASR, retrieval, grounding, refusal, injection resistance.
6. **User study:** method, participants, tasks, measures, procedure.
7. **Results:** quantitative (time, errors, SUS, trust, workload) and qualitative.
8. **Discussion:** design implications, limitations, threats to validity.
9. **Conclusion and future work**
10. **Ethics statement and artefact availability**

### Contributions (planned)

- An open-source architecture combining sensor-based context, speech and verified-docs RAG with explicit refusal.
- A design for context-adaptive output (modality, length, timing) for safety-relevant information.
- An exploratory within-subjects study (n = 8 to 12) comparing context-aware vs plain chatbot.

## Technical documentation

| Document | Content |
|---|---|
| README | Overview, quick start, status |
| docs/FEATURES.md | Requirements and acceptance criteria |
| docs/ARCHITECTURE.md | Components, data flow, security, privacy |
| docs/EVALUATION.md | Study protocol and metrics |
| API reference | Generated from FastAPI OpenAPI schema |
| Scenario guide | How to add a scenario and document pack |
| Model card | HAR model: data, splits, metrics, limitations |
| Safety case summary | Threat model, guardrails, red-team results |
| CHANGELOG | Versioned changes |

## Project results structure

```text
results/
  har/          # metrics, confusion matrices, model cards
  rag/          # retrieval, grounding, refusal, injection reports
  study/        # anonymised aggregates, figures, analysis notebooks
paper/          # LaTeX sources, figures, bib
```

Weekly progress notes (what was done, decisions, open issues) are kept in `docs/progress/` to make the final write-up easier.

## Reproducibility checklist

- [x] Fixed random seeds; seeds reported.
- [ ] Pinned dependencies (lock file) and Docker image.
- [x] Dataset download scripts with checksums (UCI HAR, WISDM); licences cited.
- [ ] Subject-wise data splits published.
- [x] Model and prompt versions recorded with every result.
- [ ] Verified document set (or a public substitute) and gold question set released.
- [x] Configs for both study conditions released.
- [x] Analysis scripts run end to end from exported logs to figures.
- [ ] Hardware used for latency numbers documented.
- [ ] Only anonymised, aggregated study data published, as covered by consent.
- [ ] Tagged release (v0.1) linked from the paper.

## Writing timeline

| Week | Output |
|---|---|
| 1 | Related-work notes, study protocol draft |
| 3 to 6 | System design section written alongside implementation |
| 7 | Methods section final before study |
| 8 | Results and figures |
| 9 | Full draft, internal review |
| 10 | Revision, camera-ready style PDF, release |
