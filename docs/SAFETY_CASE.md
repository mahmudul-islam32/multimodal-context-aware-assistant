# Safety case summary (v0.1)

## Threat model

| Threat | Mitigation | Evidence |
|---|---|---|
| Hallucinated instruction | Answers only from retrieved approved chunks; per-sentence grounding check (content words + exact numbers); unsupported sentences dropped; refusal if none remain | `tests/test_rag.py::test_llm_output_ungrounded_sentence_removed`, faithfulness in `results/REPORT.md` |
| Out-of-scope question answered | Evidence gate: term coverage, unknown part IDs, conflicting sources → refusal with escalation contact | 13/13 out-of-scope refused, 0/25 false refusals (small author-written set) |
| Direct prompt injection | Pattern-based detector before retrieval; LLM has no tools; system prompt marker leak check | 0/12 red-team attacks succeeded; 0/4 benign false positives |
| Indirect injection (document) | Scan at ingestion → quarantine; documents placed in delimited `<document>` blocks labelled as data | `forum_tip_injected.md` quarantined, probe answer clean |
| Unsafe advice (skip lockout, bypass interlock) | Output checker blocks such phrases unless they appear verbatim in sources | `OutputChecker` |
| Unapproved / outdated document | Approval metadata required; versioning retires old chunks | `vendor_bulletin_unverified.md` rejected; v2.0 manual (20 Nm) retired by v2.1 (25 Nm) |
| Wrong context → wrong modality | Hysteresis, visible context indicator, manual override, full answer always one tap away | `tests/test_context_policy.py` |

Known gaps: the injection detector is rule-based and can be bypassed by paraphrase. A trained classifier and a larger red-team set are planned. The gold and red-team sets are small and were written by the author.

## Privacy and data flow

| Data | Where processed | Stored? | Retention |
|---|---|---|---|
| Audio | in memory on the backend (Whisper) or in the browser (Web Speech) | No | – |
| Transcripts and queries | backend | Event log, **PII-redacted** (names, e-mail, phone, IBAN, dates, IDs) | study period; `purge_older_than` |
| IMU samples | backend, in a rolling buffer | No (only classification results and context changes are logged) | – |
| Camera frames | backend, in memory, downscaled to 160 × 120 | **Never**; only a 32 × 32 grey thumbnail is held for motion, dropped on disable | – |
| Study events | SQLite (`MCAA_DB`) | Yes, by pseudonym `Pxx` only | deletable per participant (`DELETE /api/study/participant/{id}`); at most 6 months |
| External LLM (optional) | EU-hosted endpoint | redacted query + chunks only | provider terms |
