# Model card: HAR activity classifier (v0.1)

| | |
|---|---|
| Models | (1) 136 hand-crafted features + random forest (200 trees, balanced class weights); (2) 1D-CNN, 3 conv blocks, ~25k parameters |
| Input | 128 × 6 window (2.56 s at 50 Hz): acceleration in g (incl. gravity), gyroscope in rad/s, 20 Hz low-pass |
| Output | probabilities over WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING, WORKING |
| Mapping to context | walking* → moving; sitting/standing/laying → idle; working → hands_busy |
| Intended use | Research prototype: picking the output modality of an assistant. **Not** for safety decisions, health monitoring or worker surveillance. |

## Training data

- **Reported v0.1 numbers: synthetic data only** (`mcaa.sensors.synthetic`). The generator simulates 12 subjects with different gait frequency, amplitude, device tilt and tremor. It is easier than real data.
- Supported real datasets: UCI HAR (30 subjects, 6 activities, 50 Hz) and WISDM 2019 (51 subjects, 18 activities, 20 Hz, resampled to 50 Hz). Download them with `python -m mcaa.data.download`. The SHA-256 of each archive is recorded.
- WORKING (hands busy) is in neither public dataset. It is learned from a short calibration recording of the target task (`FeatureModel.calibrate`, `CNNModel.calibrate`).

## Evaluation (synthetic, leave-subjects-out, 4 folds, seed 42)

| Model | Macro-F1 | Fold std | Latency per live window |
|---|---|---|---|
| Features + RF | 0.974 | 0.022 | 8.8 ms (CPU) |
| 1D-CNN | 0.900 | 0.034 | 0.3 ms (CPU) |

Calibration with 30 s per activity from a new simulated user raises task-state accuracy from 0.86 to 1.00. Confusion matrices are in `results/metrics.json` and `docs/assets/har_results.png`.

## Limitations

- No real-data results yet. Expect lower scores, especially for sitting versus standing and for wrist versus pocket placement.
- Phone placement and orientation are not normalised beyond gravity separation.
- Context decisions should stay overridable by the user (the UI makes them so).
