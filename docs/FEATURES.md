# Feature Specification: multimodal-context-aware-assistant

All features are **Planned**. Priority uses MoSCoW: Must, Should, Could.

| ID | Feature | Priority | Status |
|---|---|---|---|
| F-01 | Sensor streaming and signal preprocessing | Must | Planned |
| F-02 | Activity recognition model | Must | Planned |
| F-03 | Speech input (Whisper) | Must | Planned |
| F-04 | Context engine | Must | Planned |
| F-05 | Verified document store | Must | Planned |
| F-06 | Grounded answers with citations | Must | Planned |
| F-07 | Refusal and escalation | Must | Planned |
| F-08 | Prompt-injection and output guardrails | Must | Planned |
| F-09 | Adaptive output policy | Must | Planned |
| F-10 | Multimodal tablet UI (voice and touch) | Must | Planned |
| F-11 | Privacy and data minimisation | Must | Planned |
| F-12 | Study mode and experiment logging | Must | Planned |
| F-13 | Configurable scenarios | Should | Planned |
| F-14 | Text-to-speech output | Should | Planned |
| F-15 | Offline evaluation harness | Should | Planned |
| F-16 | Camera-based context (optional modality) | Could | Planned |

---

### F-01: Sensor streaming and signal preprocessing
**Description:** Stream accelerometer and gyroscope data from a smartphone or wearable to the backend, resample, filter and cut into fixed windows (e.g. 2.56 s with 50% overlap, as in UCI HAR).
**User story:** As a researcher, I want a clean, timestamped sensor stream, so that live data matches the format the model was trained on.
**Acceptance criteria:**
- Streams at a stable rate (target 50 Hz) over WebSocket with < 200 ms added latency.
- Resampling handles both 50 Hz (UCI HAR) and 20 Hz (WISDM) sources.
- Dropped samples are detected and reported.
**Priority:** Must · **Status:** Planned

### F-02: Activity recognition model
**Description:** Classifier (scikit-learn baseline on hand-crafted features, PyTorch 1D-CNN or small transformer) trained on UCI HAR and WISDM, then mapped to task-relevant states (idle, walking, hands busy).
**User story:** As the assistant, I want to know whether the user is moving or working with their hands, so that I can pick the right output.
**Acceptance criteria:**
- Subject-wise (leave-subjects-out) evaluation is reported, not random splits only.
- Macro-F1 and confusion matrix documented for each dataset.
- Inference < 50 ms per window on a laptop CPU.
- A short calibration recording of the target task can fine-tune the model.
**Priority:** Must · **Status:** Planned

### F-03: Speech input (Whisper)
**Description:** Local speech-to-text with Whisper or faster-whisper, push-to-talk and optional voice activity detection.
**User story:** As a technician with dirty gloves, I want to ask a question by voice, so that I don't have to touch the screen.
**Acceptance criteria:**
- Word error rate measured on a small recorded test set in the target setting (incl. background noise).
- Audio is processed locally by default and not stored unless study consent allows it.
- End-of-speech to transcript < 1.5 s for short utterances on target hardware.
**Priority:** Must · **Status:** Planned

### F-04: Context engine
**Description:** Combines activity, UI interaction, current task step, ambient noise and time since last interaction into a context state with confidence.
**User story:** As a designer, I want one explicit, inspectable context state, so that adaptation decisions are explainable.
**Acceptance criteria:**
- States and transition rules are documented and configurable.
- Hysteresis prevents state flicker (no more than one change per N seconds, configurable).
- Each answer log entry records the context state used.
**Priority:** Must · **Status:** Planned

### F-05: Verified document store
**Description:** Ingestion of an approved, versioned document set (manuals, SOPs, guidelines) with metadata: source, version, approver, date.
**User story:** As a safety officer, I want the assistant to use only documents we approved, so that answers are traceable.
**Acceptance criteria:**
- Only documents with an approval record are indexed.
- Every chunk keeps its source, page/section and version.
- Re-ingestion of a new version retires the old chunks.
**Priority:** Must · **Status:** Planned

### F-06: Grounded answers with citations
**Description:** RAG pipeline that answers only from retrieved chunks and cites them.
**User story:** As a care worker, I want to see where an instruction comes from, so that I can trust and check it.
**Acceptance criteria:**
- Every factual sentence links to at least one cited chunk.
- Grounding/faithfulness score measured on a gold question set (F-15).
- Spoken answers name the source briefly ("from the pump manual, section 4").
**Priority:** Must · **Status:** Planned

### F-07: Refusal and escalation
**Description:** When retrieval confidence is low or sources conflict, the assistant declines and suggests an escalation path.
**User story:** As a user in a critical step, I want a clear "I don't know" rather than a guess.
**Acceptance criteria:**
- Out-of-scope test questions are refused at a measured rate (target ≥ 90%).
- In-scope questions are not over-refused (false refusal rate reported).
- Refusals name the escalation contact defined in the scenario config.
**Priority:** Must · **Status:** Planned

### F-08: Prompt-injection and output guardrails
**Description:** Detection of injected instructions in user input and in documents, strict system prompt separation, output checks for unsafe or ungrounded content.
**User story:** As an operator, I want the assistant to resist manipulated input, so that it stays within its role.
**Acceptance criteria:**
- Red-team set of injection attacks (direct and in documents) with attack success rate reported.
- The LLM has no tool access beyond retrieval.
- Blocked events are logged in the audit log.
**Priority:** Must · **Status:** Planned

### F-09: Adaptive output policy
**Description:** Rules (and later optional learned policy) that choose modality, length and detail from the context state.
**User story:** As a technician holding a part, I want a one-sentence spoken answer, and the full steps on screen when I'm free.
**Acceptance criteria:**
- `HANDS_BUSY`/`MOVING`: voice first, ≤ 25 words, screen shows a summary.
- `IDLE`: full answer on screen with citations, voice optional.
- `NOISY`: screen first with a visual cue.
- Policy is a versioned config file; every decision is logged.
**Priority:** Must · **Status:** Planned

### F-10: Multimodal tablet UI (voice and touch)
**Description:** PWA for tablets with large touch targets, push-to-talk, step-by-step task view, visible context indicator and source panel.
**User story:** As a user, I want to switch between speaking and tapping at any time without losing the conversation.
**Acceptance criteria:**
- Touch targets ≥ 44×44 px; WCAG 2.1 AA as target, checked with automated tools and a manual review.
- Context state is visible to the user and can be overridden manually.
- Works on a 10-inch tablet in landscape and portrait.
**Priority:** Must · **Status:** Planned

### F-11: Privacy and data minimisation
**Description:** GDPR-oriented design: local processing by default, PII redaction before any external LLM call, retention limits, pseudonymised study data.
**User story:** As a participant, I want my voice and movement data protected and deleted after the study.
**Acceptance criteria:**
- Data flow diagram with storage locations and retention documented.
- PII redaction tested on a sample set.
- Raw audio and sensor data deletable per participant ID.
**Priority:** Must · **Status:** Planned

### F-12: Study mode and experiment logging
**Description:** Switch between condition A (context-aware) and B (plain chatbot over the same documents), with counterbalancing and timestamped event logs.
**User story:** As a researcher, I want both conditions in one system, so that only the adaptation differs.
**Acceptance criteria:**
- Condition assignment follows a counterbalanced schedule.
- Logs capture task start/end, queries, answers, errors, refusals and context state.
- Export to CSV/Parquet for analysis.
**Priority:** Must · **Status:** Planned

### F-13: Configurable scenarios
**Description:** YAML scenario packs (technician default, care worker; driver as simulator-only) defining documents, tasks, context states, escalation contacts.
**User story:** As a researcher, I want to change the setting without code changes.
**Acceptance criteria:**
- Two scenarios ship with sample (non-confidential) documents.
- Scenario schema is validated on start.
- The driver scenario is clearly marked as not for real road use.
**Priority:** Should · **Status:** Planned

### F-14: Text-to-speech output
**Description:** Local TTS (e.g. Piper) with interruptible playback.
**User story:** As a user, I want to interrupt a spoken answer by tapping or saying "stop".
**Acceptance criteria:**
- Playback starts < 1 s after answer is ready.
- Barge-in stops playback.
**Priority:** Should · **Status:** Planned

### F-15: Offline evaluation harness
**Description:** Automated tests for HAR accuracy, ASR WER, retrieval recall, grounding, refusal and injection resistance, run in CI.
**User story:** As a developer, I want regressions caught before the user study.
**Acceptance criteria:**
- One command produces a metrics report.
- Gold question set covers in-scope, out-of-scope and adversarial items.
**Priority:** Should · **Status:** Planned

### F-16: Camera-based context (optional modality)
**Description:** Optional on-device vision model (e.g. object or hand detection) as an additional context cue.
**User story:** As a technician, I want the assistant to recognise the component I'm looking at.
**Acceptance criteria:**
- Off by default; requires explicit consent.
- Frames are processed locally and never stored.
**Priority:** Could · **Status:** Planned
