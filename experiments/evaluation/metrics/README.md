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

| Scenario | Detector | Precision | Recall | F1 | False Alarm Rate | Detection Delay |
|---|---|---:|---:|---:|---:|---:|
| Sudden | DriftLens | 1.000 | 1.000 | 1.000 | 0.000 | 0 |
| Sudden | MCD-DD | 1.000 | 0.400 | 0.571 | 0.000 | 0 |
| Gradual | DriftLens | 1.000 | 0.967 | 0.983 | 0.000 | 1 |
| Gradual | MCD-DD | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| Volume | DriftLens | 0.000 | 0.000 | 0.000 | 0.022 | N/A |
| Volume | MCD-DD | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| No Drift | DriftLens | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| No Drift | MCD-DD | 0.000 | 0.000 | 0.000 | 0.022 | N/A |

### Interpretation

- Under **sudden drift**, DriftLens detects all expected drift windows, while MCD-DD detects only part of the drift period.
- Under **gradual drift**, DriftLens detects almost all expected drift windows with a one-window detection delay, while MCD-DD produces no alarms.
- Under the **volume-change** scenario, DriftLens produces one false alarm and MCD-DD produces none.
- Under **no drift**, DriftLens produces no false alarms, while MCD-DD produces one false alarm.

---

## Detector Agreement

| Scenario | Both Alarm | Both No Alarm | DriftLens Only | MCD-DD Only | Agreement |
|---|---:|---:|---:|---:|---:|
| Sudden | 12 | 16 | 18 | 0 | 60.87% |
| Gradual | 0 | 17 | 29 | 0 | 36.96% |
| Volume | 0 | 45 | 1 | 0 | 97.83% |
| No Drift | 0 | 45 | 0 | 1 | 97.83% |

The agreement results can be used by downstream visualization and explainability components to identify windows where the detectors produce the same or different decisions.

---

## Pipeline Performance

The evaluation also records processing latency and throughput.

These measurements are **pipeline-level measurements** and should not be interpreted as isolated execution times for DriftLens or MCD-DD.

| Scenario | Mean Latency (ms) | Median Latency (ms) | Mean Throughput (posts/s) | Median Throughput (posts/s) |
|---|---:|---:|---:|---:|
| Sudden | 578.22 | 542.55 | 1820.23 | 1843.17 |
| Gradual | 582.12 | 543.75 | 1804.21 | 1839.29 |
| Volume | 1066.61 | 1226.23 | 2139.24 | 2192.03 |
| No Drift | 566.35 | 524.40 | 1852.10 | 1906.95 |

CPU utilization and memory consumption were not captured separately during these runs.

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

These plots can be reused by reporting, explainability, and dashboard components.

---

## Evaluation Script

The complete evaluation is implemented in:

```text
evaluate_metrics.py
```

It reads the existing detector result files, aligns the outputs using `window_id`, applies the scenario ground truth, calculates the evaluation metrics, compares detector decisions, and generates the result tables and plots.

The evaluation can be regenerated using:

```bash
python experiments/evaluation/metrics/evaluate_metrics.py
```

---

## Detector Input Results

The evaluation reads the detector outputs generated for each scenario:

```text
experiments/results/sudden/drift_results.jsonl
experiments/results/gradual/drift_results.jsonl
experiments/results/volume/drift_results.jsonl
experiments/results/no_drift/drift_results.jsonl
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
- CPU and memory measurements were not captured in the completed runs.
- The evaluation layer does not modify Kafka, Spark, DriftLens, MCD-DD, or scenario-generation components.
