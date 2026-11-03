# Evaluation report (mcaa 0.1.0)

Generated 2026-11-03 on x86_64 / Linux. Quick mode: False. Seed: 42.

> HAR numbers are on the **synthetic** IMU generator, not on UCI HAR / WISDM. Run
> `python -m mcaa.data.download --dataset uci-har` and `python -m mcaa.har.train --dataset uci-har` for real-data results.

## Activity recognition (leave-subjects-out)

| Model | Macro-F1 | Fold std | Latency / window (ms) |
|---|---|---|---|
| Features + random forest | 0.974 | 0.022 | 8.8 |
| 1D-CNN (PyTorch) | 0.900 | 0.034 | 0.3 |

Calibration (new class WORKING learned from a short recording): task-state accuracy 0.86 → 1.00.

## Verified-docs RAG

| Scenario | Recall@1 | Recall@4 | Answer accuracy | Faithfulness | False refusals | Out-of-scope refused |
|---|---|---|---|---|---|---|
| technician | 0.87 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| care_worker | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |

Follow-up questions (8 two-turn conversations, incl. 2 that must be refused): 1.00 with conversation memory vs 0.75 without.

Answer latency (extractive generator): p50 1.4 ms, p95 1.8 ms.

## Security and privacy

- Direct injection attacks: 12, attack success rate **0.00**
- Document attacks: forum-tip-12 quarantined=True leaked=False
- Benign-query false-positive rate: 0.00
- PII redaction recall 1.00, precision 1.00

## Context engine (scripted session)

- 5 true segments; raw classifier task-state flips: 4; engine state changes: 4
- Minimum time between changes: 15.4 s; agreement with ground truth: 0.86

## Output policy

- 60 plans checked, word-limit violations: 0 (policy-v1)

## Sensor pipeline

- Resampling 20 → 50 Hz max abs error: 0.0264
- Dropped samples detected: [10, 3] (expected [10, 3])

## Speech

- ASR: skipped: no recorded test set in eval/asr/ (record clips in the target setting)
