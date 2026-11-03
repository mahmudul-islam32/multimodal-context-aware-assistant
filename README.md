# Multimodal Context-Aware Assistant

![status](https://img.shields.io/badge/status-v0.1%20prototype-blue)
![tests](https://img.shields.io/badge/tests-42%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.11%2B-informational)

A Python research prototype of an assistant that gives **safe, short, situation-appropriate information** to people whose hands and attention are busy: a field technician repairing a machine, a care worker on a ward round, or (in a simulator only) a driver. It combines **voice, touch/text and a motion-sensor stream** (phone or wearable IMU, optionally a camera), infers what the user is doing, and answers questions **only from approved, versioned documents**: every sentence is cited, and the assistant refuses when the evidence is missing. Output adapts to context: one short spoken sentence while the user is working, the full answer on screen when they are idle.

![Demo: the assistant switches from IDLE to HANDS BUSY to MOVING as the (simulated) wrist sensor changes, and adapts the answer each time](docs/assets/ui_demo.gif)

<sub>Demo recorded headless with Playwright against the real backend; the IMU stream comes from the built-in sensor simulator. [Full-resolution MP4](docs/assets/ui_demo.mp4).</sub>

> **Status: v0.1, all 16 features in [FEATURES.md](docs/FEATURES.md) are implemented and tested.**
> Things that are **not** done yet, so the numbers below should be read with them in mind:
> - The public HAR datasets (UCI HAR, WISDM) could not be downloaded in the build environment. Loaders and a download script are included, but **every HAR number here is from the built-in synthetic IMU generator**.
> - Whisper ASR and the LLM generator are implemented but were not run against real models here (no model downloads available). The default generator is extractive and offline. ASR word error rate needs recordings from the target setting.
> - The **user study has not been run**. The study figure below uses **simulated** participants to check the analysis pipeline. It is not a result.

---

## Contents

- [What it does](#what-it-does)
- [The UI](#the-ui)
- [How it works](#how-it-works)
- [Results (offline evaluation)](#results-offline-evaluation)
- [Quick start](#quick-start)
- [Configuration](#configuration-scenarios-context-rules-policy)
- [Study mode](#study-mode)
- [Repository layout](#repository-layout)
- [Feature status](#feature-status)

## What it does

| Situation (context state) | What the user gets |
|---|---|
| `IDLE`: standing or sitting, hands free | Full answer on screen with numbered citations and the source panel |
| `HANDS_BUSY`: manual work detected, or a manual step with no touch for 20 s | **Voice first**, at most 25 words, naming the source ("From the Pump P-101 maintenance manual, Section 4.3"); a summary on screen |
| `MOVING`: walking detected | Voice first, at most 15 words; details deferred to the screen |
| `NOISY`: loud environment or poor speech recognition | **Screen first** with a flashing visual cue, no voice |
| Out of scope, conflicting sources, unknown part number | *"I can't confirm this from the approved documents … Please ask the shift supervisor (ext. 4410)."* |
| Prompt-injection attempt | Blocked before retrieval and logged in the audit log |

The same question, *"What torque do I use for the gland follower bolts?"*, in two contexts:

| `IDLE`: full answer on screen | `HANDS_BUSY`: spoken one-liner with the source named |
|---|---|
| ![idle](docs/assets/ui_idle_full_answer.png) | ![hands busy](docs/assets/ui_hands_busy_voice_first.png) |

## The UI

A tablet PWA (plain HTML/JS, served by FastAPI) with touch targets of at least 48 px, push-to-talk, a step-by-step task view, a visible context indicator that the user can override, and a source panel.

| Refusal with escalation | Injection blocked |
|---|---|
| ![refusal](docs/assets/ui_refusal_escalation.png) | ![injection](docs/assets/ui_injection_blocked.png) |
| **`NOISY`: screen first, visual cue, no voice** | **Study condition B: plain chatbot, no adaptation** |
| ![noisy](docs/assets/ui_noisy_visual_cue.png) | ![plain](docs/assets/ui_condition_b_plain.png) |

| Researcher drawer (study mode, simulator, camera consent, export) | Phone sensor bridge (`/sensor.html`) |
|---|---|
| ![study mode](docs/assets/ui_study_mode.png) | <img src="docs/assets/ui_phone_sensor_bridge.png" width="260" alt="phone sensor page"> |

## How it works

```mermaid
flowchart LR
    subgraph Inputs
        MIC[Microphone]
        TOUCH[Touch and text UI]
        IMU[Motion sensor<br/>phone or wearable]
        CAM[Camera, optional,<br/>consent required]
    end
    MIC --> ASR[Speech to text<br/>faster-whisper / Whisper / Web Speech]
    IMU --> SIG[Signal pipeline<br/>resample, filter, window]
    CAM -.-> VIS[Hands-in-view cue]
    SIG --> HAR[Activity classifier<br/>RF on features or 1D-CNN]
    HAR --> CTX[Context engine<br/>YAML rules + hysteresis]
    VIS -.-> CTX
    TOUCH --> CTX
    ASR --> GIN[Input guardrails<br/>injection scan, PII redaction]
    TOUCH --> GIN
    GIN --> RET[Hybrid retriever<br/>BM25 + TF-IDF over approved chunks]
    RET --> GATE{Evidence gate<br/>coverage, part IDs, conflicts}
    GATE -- weak --> REF[Refusal + escalation contact]
    GATE -- ok --> GEN[Generator<br/>extractive or LLM, cites n]
    GEN --> GRD[Grounding check per sentence<br/>+ output safety check]
    GRD --> POL[Adaptive output policy]
    CTX --> POL
    POL --> TTS[Voice]
    POL --> SCREEN[Screen]
    GIN & GRD & POL --> LOG[(Audit and study log, SQLite)]
```

### 1. Sensor pipeline (F-01)

Phone samples arrive over WebSocket with irregular timestamps. The pipeline detects and reports dropped samples, resamples onto a uniform 50 Hz grid (so 20 Hz WISDM data and 50 Hz UCI HAR data share one format), applies a zero-phase Butterworth low-pass filter, splits gravity from body acceleration, and cuts 2.56 s windows with 50 % overlap, the UCI HAR convention.

![signal pipeline](docs/assets/signal_pipeline.png)

### 2. Activity recognition (F-02)

Two models are evaluated **leave-subjects-out**, so no subject appears in both train and test:

- **Features + random forest** (scikit-learn): 136 time- and frequency-domain features per window.
- **1D-CNN** (PyTorch): three conv blocks on the raw filtered window.

The public datasets have no "hands busy" class, so the model is **calibrated** on a short recording of the target task. The new `WORKING` class is added and up-weighted (RF), or the CNN head is extended and fine-tuned with the early layers frozen.

![HAR results](docs/assets/har_results.png)

### 3. Context engine (F-04)

Activity probabilities are smoothed (EMA) and grouped into *idle / moving / hands busy*. They are then combined with touch recency, the current task step, noise level, ASR confidence and the optional camera cue through declarative rules in [`configs/context.yaml`](configs/context.yaml). **Hysteresis** debounces the result: a new state must win two evaluations in a row, and at most one change is allowed per 4 s. A manual override from the UI always wins. Every answer logs the state it used.

![context engine animation](docs/assets/context_engine.gif)

![context timeline](docs/assets/context_timeline.png)

### 4. Verified-docs RAG with refusal (F-05 to F-08)

- **Ingestion** accepts only Markdown with an approval record (`approved_by`, `approved_on`, `version`, `source`). A new version retires the old chunks. Documents containing injection markers are **quarantined**. The technician pack ships with both cases: `inbox/vendor_bulletin_unverified.md` is rejected, and `inbox/forum_tip_injected.md` is quarantined.
- **Retrieval** is hybrid: BM25 and TF-IDF cosine similarity, fused with reciprocal rank fusion. Each chunk keeps its document, version and section.
- The **evidence gate** refuses an answer when:
  - fewer than half of the query terms appear in the best chunk;
  - the question names a part or model ID (such as `P-205`) that no approved document mentions;
  - two documents give different numbers for the same thing.
- **Generation**: the default is an extractive generator. It is offline and deterministic, and it can only quote approved text. An OpenAI-compatible LLM backend (Ollama, vLLM, or an EU-hosted API) is optional. The LLM receives only the PII-redacted query and the retrieved chunks inside delimited `<document>` blocks, and it has no tools.
- **Grounding check**: every sentence must cite a chunk that contains most of its content words *and all of its numbers*. Unsupported sentences are removed. If none remain, the assistant refuses. An output checker blocks unsafe instructions (for example "bypass the interlock" or "skip lockout") that do not appear in the sources.

### 5. Output policy, speech, camera (F-03, F-09, F-14, F-16)

- **Policy** ([`configs/policy.yaml`](configs/policy.yaml), versioned and logged): maps each state to modality, voice word limit, screen detail and citation style.
- **Speech in**: push-to-talk records PCM in the browser and sends WAV to `/api/speech/transcribe`. There, faster-whisper or Whisper transcribes it locally, in memory, after energy-based VAD. If neither is installed, the browser's Web Speech API is the fallback. Saying "stop" triggers barge-in.
- **Speech out**: browser `speechSynthesis`, or Piper on the server. Tapping the answer, pressing **Stop**, or saying "stop" interrupts playback.
- **Camera** (off by default): requires explicit consent. Frames are downscaled and processed in memory and never stored. A skin-tone and motion heuristic yields a *hands in view* cue, and a stronger detector can be plugged in.

## Results (offline evaluation)

`mcaa-eval` runs everything below and writes [`results/REPORT.md`](results/REPORT.md) and `results/metrics.json`. CI runs a quick version on every push.

![RAG and refusal results](docs/assets/rag_safety.png)

| Component | Metric | Result |
|---|---|---|
| HAR, features + RF (synthetic, 12 subjects, LOSO) | macro-F1 · latency per live window | **0.974** · 11.4 ms |
| HAR, 1D-CNN (synthetic, LOSO) | macro-F1 · latency per live window | **0.906** · 0.4 ms |
| HAR calibration (30 s per activity) | task-state accuracy on a new user | 0.86 → **1.00** |
| Retrieval, technician / care | recall@1 · recall@4 | 0.87 / 1.00 · 1.00 / 1.00 |
| Answers (25 in-scope gold questions) | key fact present · faithfulness | 1.00 · 1.00 |
| Refusal | out-of-scope refused · false refusals | 13/13 · 0/25 |
| Prompt injection | 12 direct attacks succeeded · document attack | **0/12** · quarantined, nothing leaked |
| Guardrail false positives | benign "ignore / system / lockout" queries flagged | 0/4 |
| PII redaction | recall · precision (5 labelled samples) | 1.00 · 1.00 |
| Context engine (scripted 115 s session) | state changes for 5 true segments · min gap · agreement | 4 · 15.4 s · 0.86 (the gap is detection lag) |
| Output policy | voice word-limit violations (60 plans) | 0 |
| Sensor pipeline | 20→50 Hz resampling max error · drops detected | 0.026 g · 10 and 3 of 10 and 3 |
| Speech | WER | not measured yet (needs recordings in the target setting) |

How to read these numbers:

- **The gold set is small and was written by the author against the same documents**, so the 1.00 scores show the pipeline works as designed, not that it generalises. The right panel above shows that the technician refusal margin is thin. One out-of-scope question (*"How do I calibrate the flow meter FT-330?"*) reaches the 0.5 coverage threshold and is refused only because of the unknown part-ID check. A larger, independently written question set is the next step.
- **HAR on synthetic data is easy compared with real data.** Expect lower scores on UCI HAR and WISDM, especially for sitting versus standing.
- **The context engine debounces flicker by design.** The cost is about 3 to 4 s of lag after each real transition, which explains the 0.86 agreement with ground truth.

### Study analysis pipeline (simulated data)

[`eval/analysis/analyze_study.py`](eval/analysis/analyze_study.py) implements the [analysis plan](docs/EVALUATION.md): median and IQR, a Shapiro-Wilk check on the paired differences, then a paired t-test or a Wilcoxon signed-rank test, Cohen's d_z or rank-biserial r with bootstrap 95 % CIs, and an order-effect check. To make sure it runs end to end, it can generate **simulated** participants:

![simulated paired comparison](docs/assets/simulated_paired_comparison.png)

## Quick start

```bash
git clone https://github.com/mahmudul-islam32/multimodal-context-aware-assistant.git
cd multimodal-context-aware-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ,torch for the CNN; ,speech for faster-whisper
```

Run the app. Open <http://localhost:8000> on a tablet or laptop and `http://<your-ip>:8000/sensor.html` on a phone. Browsers expose phone motion sensors only over HTTPS or on localhost.

```bash
mcaa-api
```

No phone? Open **Researcher → Simulated sensor** and click *working*, *walking* or *standing*.

Other commands:

```bash
pytest -q                                           # 42 tests
```

```bash
mcaa-eval                                           # full offline report -> results/REPORT.md
```

```bash
python -m mcaa.har.train --config configs/har.yaml  # train + LOSO-evaluate HAR, saves models/har_rf.pkl
```

```bash
python -m mcaa.data.download --dataset uci-har      # real data (also: wisdm), records SHA-256
```

```bash
python -m mcaa.har.train --dataset uci-har --model cnn
```

```bash
python scripts/make_figures.py                      # regenerate figures in docs/assets
```

```bash
python scripts/record_demo.py                       # regenerate UI screenshots and demo video (Playwright + ffmpeg)
```

Docker, optionally with a local LLM:

```bash
docker compose up                                   # API + UI on :8000
```

```bash
docker compose --profile llm up                     # with Ollama; set MCAA_GENERATOR=llm
```

| Environment variable | Default | Meaning |
|---|---|---|
| `MCAA_SCENARIO` | `technician` | `technician`, `care_worker`, `driver_sim` |
| `MCAA_GENERATOR` | `extractive` | `llm` uses an OpenAI-compatible endpoint |
| `MCAA_LLM_BASE_URL` / `MCAA_LLM_MODEL` / `MCAA_LLM_API_KEY` | Ollama on localhost | LLM endpoint |
| `MCAA_DB` | in memory | SQLite event log path |
| `MCAA_HAR_MODEL` | `models/har_rf.pkl` | if missing, a model is trained on synthetic data at start-up |

## Configuration: scenarios, context rules, policy

Scenarios are YAML packs, validated with pydantic at start-up ([docs/SCENARIOS.md](docs/SCENARIOS.md)). Each pack defines the document folder, two comparable task sets (A and B, for counterbalancing), which steps are manual, and the escalation contact. Three packs ship:

- **technician** (default): pump seal replacement and bearing maintenance; manual and LOTO procedure.
- **care_worker**: repositioning and observations; hand hygiene, repositioning and vital-signs guidelines.
- **driver_sim**: `simulator_only: true`. Validation fails unless the pack carries a warning, which the UI shows as a permanent banner.

All sample documents are **fictional** and written for this project. They are not real manuals or clinical guidance.

## Study mode

Condition **A** is the context-aware assistant. Condition **B** is a plain chatbot that uses the *same* documents, retrieval and guardrails but has no adaptation and no context panel. Participants are pseudonyms (`P01`); the API rejects names. Each participant gets a balanced Latin-square schedule of condition order × task-set pairing.

The SQLite log records task start and end, steps, queries (PII-redacted), answers (with context state, policy version, modality, citations and refusals), errors, overrides, context changes and guardrail events.

- **Export:** `GET /api/study/export?fmt=csv|parquet` writes event-level and per-task tables.
- **Deletion:** `DELETE /api/study/participant/P07` removes all of that participant's data.
- **Retention:** `EventLogger.purge_older_than(days)` enforces the retention limit.

## Repository layout

```text
src/mcaa/
  sensors/     preprocess.py (resample, filter, windows, drops, StreamBuffer), synthetic.py
  har/         features.py, models.py (RF, LOSO, calibration), cnn.py (PyTorch), datasets.py, train.py
  speech/      asr.py (faster-whisper / Whisper), vad.py, wer.py, tts.py (Piper, barge-in)
  context/     engine.py (rules, EMA, hysteresis, override)
  rag/         store.py, retriever.py, generator.py, grounding.py, guardrails.py, pii.py, pipeline.py
  policy/      policy.py
  vision/      camera.py (consent-gated, never stores frames)
  study/       conditions.py (counterbalancing), logger.py (SQLite, export, deletion)
  api/app.py   FastAPI: REST + /ws/ui + /ws/sensors + static PWA
  web/static/  index.html, app.js, style.css, sensor.html, manifest, service worker
  assistant.py orchestrator    evaluation.py  offline harness (mcaa-eval)
configs/       scenarios/*.yaml, context.yaml, policy.yaml, har.yaml
data/docs/     approved sample documents per scenario (+ inbox/ with rejected and quarantined examples)
eval/          gold_questions.yaml, redteam.yaml, analysis/analyze_study.py
scripts/       make_figures.py, record_demo.py
results/       REPORT.md, metrics.json
tests/         42 tests, including API and WebSocket tests
```

## Feature status

| ID | Feature | Status | Where |
|---|---|---|---|
| F-01 | Sensor streaming and preprocessing | ✅ | `sensors/`, `/ws/sensors`, `sensor.html` |
| F-02 | Activity recognition (RF + 1D-CNN, LOSO, calibration) | ✅ synthetic data only | `har/` |
| F-03 | Speech input | ✅ implemented; WER not yet measured | `speech/asr.py`, push-to-talk |
| F-04 | Context engine | ✅ | `context/`, `configs/context.yaml` |
| F-05 | Verified document store | ✅ | `rag/store.py` |
| F-06 | Grounded answers with citations | ✅ | `rag/generator.py`, `rag/grounding.py` |
| F-07 | Refusal and escalation | ✅ | `rag/pipeline.py` |
| F-08 | Injection and output guardrails | ✅ | `rag/guardrails.py` |
| F-09 | Adaptive output policy | ✅ | `policy/`, `configs/policy.yaml` |
| F-10 | Multimodal tablet UI | ✅ static PWA instead of Next.js; no formal WCAG audit yet | `web/static/` |
| F-11 | Privacy and data minimisation | ✅ | `rag/pii.py`, `study/logger.py`, [docs/SAFETY_CASE.md](docs/SAFETY_CASE.md) |
| F-12 | Study mode and logging | ✅ | `study/`, researcher drawer |
| F-13 | Configurable scenarios | ✅ | `configs/scenarios/` |
| F-14 | Text-to-speech with barge-in | ✅ | `speech/tts.py`, browser TTS |
| F-15 | Offline evaluation harness | ✅ | `evaluation.py`, CI |
| F-16 | Camera context (optional) | ✅ heuristic cue | `vision/camera.py` |

More documentation: [Features](docs/FEATURES.md) · [Architecture](docs/ARCHITECTURE.md) · [Evaluation and user study](docs/EVALUATION.md) · [Reporting](docs/REPORTING.md) · [HAR model card](docs/MODEL_CARD.md) · [Scenario guide](docs/SCENARIOS.md) · [Safety case](docs/SAFETY_CASE.md) · [Changelog](CHANGELOG.md)

## Contributing

Issues with feedback on the features, the study design or the safety approach are welcome. Please open an issue before a large pull request, run `ruff check` and `pytest`, and add tests.

## Author

Mahmudul Islam, M.Sc. Computer Science, backend and full-stack engineer with LLM/RAG systems in production at CIVIS.digital. GitHub: [@mahmudul-islam32](https://github.com/mahmudul-islam32)

## License

[MIT](LICENSE). Datasets: UCI HAR and WISDM are CC BY 4.0. Cite them when you use the download scripts.
