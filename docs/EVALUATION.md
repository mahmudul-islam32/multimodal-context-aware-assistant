# Evaluation and User Study Protocol

> Status: design. The study has **not** been run. Everything here is **planned**.

## Research questions

- **RQ1:** Does the context-aware assistant reduce task completion time compared with a plain chatbot over the same verified documents?
- **RQ2:** Does it reduce task errors?
- **RQ3:** How does it affect usability (SUS), perceived workload and trust?
- **RQ4 (technical):** How accurate are the components (activity recognition, ASR, retrieval, grounding, refusal, injection resistance)?

## Hypotheses

- H1: Task time is lower in condition A (context-aware) than in B (plain chatbot).
- H2: Errors are fewer in A than in B.
- H3: SUS and trust scores are higher in A than in B.

## Design

- **Within-subjects**, two conditions, counterbalanced order (AB / BA).
- **Condition A:** context-aware (sensor + voice + adaptive output policy).
- **Condition B:** plain chatbot: same documents, retrieval, LLM and guardrails, text UI, no adaptation.
- **Participants:** 8 to 12 adults, recruited from colleagues and university contacts; target setting experience recorded but not required.
- **Tasks:** two comparable task sets (e.g. a mock device maintenance procedure with 6 to 8 steps for the technician scenario, or a mock care checklist), assigned in counterbalanced order so each condition is paired with each task set.
- **Session:** ~60 minutes: consent, briefing, 5-minute training per condition, task, questionnaires, short interview.
- **Pilot:** 2 people before the main study to fix tasks and timing.

## Metrics

| Metric | Measure | Source |
|---|---|---|
| Task completion time | Seconds from task start to final step | System log + observer |
| Error rate | Wrong / skipped / unsafe steps per task, coded with a predefined error list | Observer sheet, video if consented |
| Usability | System Usability Scale (SUS, 10 items, 0 to 100 score) | Questionnaire |
| Workload | NASA-TLX (raw) | Questionnaire |
| Trust | Trust in automation scale (e.g. Jian et al., 2000) or a short validated alternative | Questionnaire |
| Interaction | Queries, modality used, refusals, manual context overrides | System log |
| Answer quality | Grounded answers, unsupported claims | Log review |
| Preference and qualitative feedback | Semi-structured interview | Notes / transcript |

Component metrics (offline, RQ4): HAR macro-F1 with leave-subjects-out splits; ASR WER on a recorded set; retrieval recall@k; faithfulness of answers; refusal rate on out-of-scope questions and false refusal rate; injection attack success rate.

## Analysis plan

- Small sample: report descriptive statistics (median, IQR) and effect sizes with confidence intervals first.
- Paired comparisons: paired t-test if differences look approximately normal, otherwise Wilcoxon signed-rank; effect size (Cohen's d_z or rank-biserial r).
- Check order effects (condition × order).
- Qualitative data: thematic analysis of interview notes.
- Results are presented as exploratory; no claims of generalisation beyond the sample.
- Analysis scripts in `eval/analysis/`, run on the exported, pseudonymised logs.

## Ethics, consent and data protection

- Check with the host institution whether ethics committee review is needed; follow its process if yes.
- Written informed consent covering: purpose, procedure, recorded data (sensor, audio, video), voluntary participation, right to withdraw at any time without giving reasons, retention, and deletion.
- Separate opt-in for audio and video recording.
- No real patient data and no real safety-critical equipment; tasks are mock procedures. The driver scenario is simulator only.
- Data pseudonymised by participant ID; key file stored separately; raw recordings deleted after analysis (planned: at most 6 months).
- Processing in line with GDPR; data stored on institution or local machines in the EU.

## Threats to validity

- Small sample and lab setting limit generalisation.
- Novelty effect for voice/sensor interaction.
- Experimenter bias (the author built the system): use a scripted briefing and predefined error coding.
- Task set difficulty differences: balanced via counterbalancing and pilot testing.
