# Changelog

## 0.2.0 (2026-11-03)

- **Follow-up questions:** short follow-ups ("and during the day?", "what about the guard screws?") are expanded with the previous question's topic. The evidence gate checks the user's own words, so a follow-up about an unknown part is still refused. The UI shows how the question was interpreted.
- **Proactive step briefings:** when a step starts, the assistant retrieves the approved text for it and surfaces the key safety sentence (condition A only). It goes through the same gate and grounding check, and the output policy decides whether it is spoken.
- **Document governance workflow:** upload → injection scan (quarantine) → diff against the active version → approval by a second person (four-eyes rule) → indexing and retirement of the old version. Approvals persist across restarts. Admin page: `/admin.html`.
- **Analytics dashboard** (`/dashboard.html`): KPIs, context timeline, answers by context and modality, guardrail events, refusal reasons, top sources, recent answers. Data from `GET /api/analytics`.
- **Answer feedback:** 👍/👎 per answer (`POST /api/feedback`), logged and shown in the dashboard.
- **Operations:** Prometheus `GET /metrics` (answers, latency, guardrail events, context changes, HAR inference time, document actions); `MCAA_ADMIN_TOKEN` protects the governance, analytics, export and deletion endpoints; `/api/health` reports version, uptime, index size and auth mode.
- Evaluation: two-turn follow-up conversations added to the gold set.
- Default HAR model is cached per process (faster scenario switches and tests).

## 0.1.0 (2026-11-03)

First runnable version. Implements all 16 features in `docs/FEATURES.md`:

- Sensor pipeline, synthetic IMU generator, UCI HAR / WISDM loaders and download script
- HAR: feature + random forest baseline and PyTorch 1D-CNN, leave-subjects-out evaluation, calibration
- Speech: faster-whisper / Whisper backends, energy VAD, WER, TTS with barge-in
- Context engine with YAML rules, EMA smoothing, hysteresis and manual override
- Verified document store with approval and versioning, hybrid retriever, extractive and LLM generators, grounding check, refusal with escalation, injection and output guardrails, PII redaction
- Adaptive output policy (versioned YAML)
- FastAPI backend with WebSockets, tablet PWA, phone sensor bridge, optional camera context
- Study mode: counterbalanced schedule, SQLite event log, CSV/Parquet export, per-participant deletion, analysis script
- Three scenario packs (technician, care worker, driver simulator only)
- Offline evaluation harness (`mcaa-eval`), 42 tests, CI, Docker
