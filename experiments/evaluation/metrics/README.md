# Drift Detection Evaluation

## Overview

This module provides the evaluation and comparison layer for the concept drift detection pipeline.

It combines the outputs of the two implemented drift detectors, **DriftLens** and **MCD-DD**, using `window_id` and evaluates their detection behavior against the known ground truth from the controlled experimental scenarios.

The module is designed to provide reusable results for further analysis, visualization, explainability, and dashboard integration.

---

## Experimental Scenarios

The evaluation currently covers four scenarios:

- **Sudden Drift**  
  Concept drift begins at position 30.

- **Gradual Drift**  
  Concept drift begins at position 30 and transitions until position 45.

- **Volume Change**  
  Stream volume increases by a factor of 3 at position 30 without a corresponding concept drift.

- **No Drift**  
  No concept drift or volume change is introduced.

The evaluation uses **14 reference windows**, followed by **46 non-reference windows** used for detector evaluation.

---

## Evaluation Metrics

The module calculates the following detection metrics:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Precision
- Recall
- F1-score
- False Alarm Rate
- Detection Delay

It also compares the two detectors based on:

- Alarm agreement
- Alarm disagreement
- Detector-specific alarms
- Common alarms
- Common non-alarm windows
- Detector scores across windows

---

## Detection Results

Regenerated on 2026-10-06 from the current detector outputs (`output/experiments/<scenario>/scores.jsonl`, fixed MCD-DD, model seed 42, RTX 5060 GPU). The earlier version of this table was computed from an older MCD-DD run.

| Scenario | Detector | Precision | Recall | F1 | False Alarm Rate | Detection Delay |
|---|---|---:|---:|---:|---:|---:|
| Sudden | DriftLens | 1.000 | 1.000 | 1.000 | 0.000 | 0 |
| Sudden | MCD-DD | 1.000 | 0.300 | 0.462 | 0.000 | 1 |
| Gradual | DriftLens | 1.000 | 0.967 | 0.983 | 0.000 | 1 |
| Gradual | MCD-DD | 1.000 | 0.033 | 0.065 | 0.000 | 14 |
| Volume | DriftLens | 0.000 | 0.000 | 0.000 | 0.022 | N/A |
| Volume | MCD-DD | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| No Drift | DriftLens | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| No Drift | MCD-DD | 0.000 | 0.000 | 0.000 | 0.000 | N/A |

### Interpretation

- Under **sudden drift**, both detectors catch the change: DriftLens at window 30 (delay 0), MCD-DD at window 31 (delay 1). Neither raises a false alarm before the change.
- Under **gradual drift**, DriftLens alarms from window 31 (delay 1). MCD-DD alarms once, at window 44 (delay 14). Its score rises through the transition but crosses the threshold only near the end (see `gradual_score_comparison.png`).
- Under the **volume-change** scenario, DriftLens produces one false alarm and MCD-DD produces none.
- Under **no drift**, neither detector raises an alarm.

**How to read MCD-DD's recall.** Recall here counts *every* window from the drift start to the end of the stream as one that should alarm. That fits DriftLens, which compares each window with the fixed reference and keeps alarming. MCD-DD is a change-point detector: it compares each window with the previous 10 windows, so it alarms for about 10 windows after a change (windows 31-39 in `sudden`) and then goes quiet once the new concept is the norm. Its low recall reflects that design. Detection delay and false alarms are the fair comparison between the two.

**MCD-DD varies with its random seed.** Over 5 model seeds, MCD-DD detected sudden drift in 5/5 runs (delay 0-2) and gradual drift in 4/5 runs (delay 6-14). Seed 42, used here, is the slowest on gradual. See `experiments/results/mcddd_seeds_trained.md`.

---

## Detector Agreement

| Scenario | Both Alarm | Both No Alarm | DriftLens Only | MCD-DD Only | Agreement |
|---|---:|---:|---:|---:|---:|
| Sudden | 9 | 16 | 21 | 0 | 54.35% |
| Gradual | 1 | 17 | 28 | 0 | 39.13% |
| Volume | 0 | 45 | 1 | 0 | 97.83% |
| No Drift | 0 | 46 | 0 | 0 | 100.00% |

Every MCD-DD alarm is also a DriftLens alarm (MCD-DD Only = 0). Most disagreements are windows long after the change, where DriftLens still alarms and MCD-DD has adapted.

The agreement results can be used by downstream visualization and explainability components to identify windows where the detectors produce the same or different decisions.

---

## Pipeline Performance

The evaluation also records processing latency and throughput.

These measurements are **pipeline-level measurements**: per window, they cover embedding the tweets (MiniLM on the GPU, no cache) plus scoring with both detectors. They should not be read as isolated execution times for DriftLens or MCD-DD.

| Scenario | Mean Latency (ms) | Median Latency (ms) | Mean Throughput (posts/s) | Median Throughput (posts/s) |
|---|---:|---:|---:|---:|
| Sudden | 309.31 | 284.77 | 3431.50 | 3511.73 |
| Gradual | 313.60 | 286.10 | 3419.38 | 3495.30 |
| Volume | 639.73 | 776.23 | 3626.28 | 3762.83 |
| No Drift | 297.95 | 275.24 | 3549.49 | 3633.39 |

Measured on an RTX 5060 laptop GPU with torch 2.11+cu128. Volume windows hold 3000 tweets after position 30, hence the higher latency. CPU utilization and memory consumption were not captured separately during these runs.

---

## Output Files

The evaluation results are stored in:

```text
experiments/evaluation/metrics/
```

### Evaluation files

```text
metrics.csv
```

Contains the detection metrics for each detector and scenario.

```text
comparison_summary.csv
```

Contains the main side-by-side comparison of DriftLens and MCD-DD.

```text
detector_agreement.csv
```

Contains common alarms, common non-alarms, detector-specific alarms, and agreement/disagreement rates.

```text
performance.csv
```

Contains pipeline-level latency and throughput measurements.

```text
window_comparison.csv
```

Contains window-level detector comparison information and can be used for detailed analysis or visualization.

```text
evaluation_summary.json
```

Contains the consolidated evaluation information in JSON format.

---

## Visualization Outputs

The `plots/` directory contains:

```text
plots/
├── alarm_comparison.png
├── sudden_score_comparison.png
├── gradual_score_comparison.png
├── volume_score_comparison.png
├── no_drift_score_comparison.png
├── latency_comparison.png
└── throughput_comparison.png
```

The `*_score_comparison.png` plots show **score / threshold** for each detector, with the alarm line at 1. Raw scores can't share one axis (DriftLens ≈ 0.02-0.06, MCD-DD ≈ 0.5-1.5). The raw scores and thresholds are still in `window_comparison.csv`.

These plots can be reused by reporting, explainability, and dashboard components.

---

## Evaluation Script

The complete evaluation is implemented in:

```text
evaluate_metrics.py
```

It reads the detector result files, takes the ground truth from each `experiments/scenarios/<scenario>/scenario.json`, calculates the evaluation metrics, compares detector decisions, and generates the result tables and plots.

The evaluation can be regenerated using:

```bash
python experiments/evaluation/metrics/evaluate_metrics.py
```

---

## Detector Input Results

By default, the evaluation reads the detector outputs that `drift/run_drift.py` writes for each scenario (the `scores_path` in each scenario's `config.yaml`):

```text
output/experiments/sudden/scores.jsonl
output/experiments/gradual/scores.jsonl
output/experiments/volume/scores.jsonl
output/experiments/no_drift/scores.jsonl
```

To produce them:

```bash
for s in sudden gradual volume no_drift; do python drift/run_drift.py --config experiments/scenarios/$s/config.yaml --refit; done
```

To evaluate results stored elsewhere (e.g. downloaded from Kaggle as `DIR/<scenario>/drift_results.jsonl`):

```bash
python experiments/evaluation/metrics/evaluate_metrics.py --input-dir DIR
```

The detector implementations themselves are not modified by this evaluation module.

---

## Use for Visualization and Explainability

The generated CSV and JSON files provide structured outputs that can be consumed by downstream components.

For example, dashboard or explainability components can use:

- `window_comparison.csv` for window-level detector behavior
- `metrics.csv` for performance summaries
- `detector_agreement.csv` for agreement/disagreement analysis
- `comparison_summary.csv` for side-by-side detector comparison
- `performance.csv` for latency and throughput visualization
- `evaluation_summary.json` for programmatic access to consolidated results
- files in `plots/` for static visual reporting

This separation allows the evaluation results to be reused without rerunning the underlying detector pipeline.

---

## Important Notes

- The evaluation uses the controlled ground truth defined for each scenario.
- Detection delay is measured relative to the known drift start position.
- Detection delay is not applicable to scenarios without concept drift.
- Pipeline latency and throughput are not detector-specific timings.
- Window-level recall favours detectors that keep alarming after a drift (DriftLens). For MCD-DD, use detection delay and false alarm rate (see "How to read MCD-DD's recall").
- Requires `matplotlib` and `pyyaml` (`pip install matplotlib pyyaml`).
- CPU and memory measurements were not captured in the completed runs.
- The evaluation layer does not modify Kafka, Spark, DriftLens, MCD-DD, or scenario-generation components.
