# Changelog

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
