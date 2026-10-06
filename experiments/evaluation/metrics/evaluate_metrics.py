"""Person 3 evaluation: DriftLens vs MCD-DD metrics, agreement, performance and plots.

Reads each scenario's detector output (the scores_path in experiments/scenarios/<name>/config.yaml,
i.e. output/experiments/<name>/scores.jsonl written by drift/run_drift.py) and its ground truth
(experiments/scenarios/<name>/scenario.json). Writes CSVs, JSON and plots to experiments/evaluation/metrics/.

Run (from anywhere):
    python experiments/evaluation/metrics/evaluate_metrics.py
    python experiments/evaluation/metrics/evaluate_metrics.py --input-dir DIR   # DIR/<scenario>/drift_results.jsonl
"""
import argparse
import json
import os
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files only; no display needed
import matplotlib.pyplot as plt
import yaml


# ============================================================
# Configuration
# ============================================================

SCENARIOS = ["sudden", "gradual", "volume", "no_drift"]
DETECTORS = ["driftlens", "mcddd"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "experiments"
SCENARIOS_DIR = BASE / "scenarios"
OUT_DIR = BASE / "evaluation" / "metrics"
PLOTS_DIR = OUT_DIR / "plots"

REFERENCE_WINDOWS = 14

# Optional override: DIR/<scenario>/drift_results.jsonl (e.g. results downloaded from Kaggle).
INPUT_DIR = None


def load_ground_truth(scenario):
    """Ground truth from Person 2's scenario.json (positions are 0-based window indices)."""
    with open(SCENARIOS_DIR / scenario / "scenario.json", encoding="utf-8") as f:
        gt = json.load(f)["ground_truth"]

    if gt["concept_drift"]:
        return {
            "concept_drift": True,
            "drift_start": gt["drift_start"]["position"],
            "drift_end": gt["drift_end"]["position"],
        }

    vol = gt.get("volume_change")
    return {
        "concept_drift": False,
        "volume_change": vol["position"] if vol else None,
    }


GROUND_TRUTH = {s: load_ground_truth(s) for s in SCENARIOS}


# ============================================================
# Utility functions
# ============================================================

def results_path(scenario):
    if INPUT_DIR is not None:
        return INPUT_DIR / scenario / "drift_results.jsonl"

    # Same file drift/run_drift.py writes for this scenario.
    with open(SCENARIOS_DIR / scenario / "config.yaml", encoding="utf-8") as f:
        scores = yaml.safe_load(f)["output"]["scores_path"]
    return PROJECT_ROOT / scores


def load_results(scenario):
    path = results_path(scenario)

    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def detector_alarm(row, detector):
    return bool(row.get(f"{detector}_alarm") is True)


def threshold_ratio(row, detector):
    """score / threshold: > 1 means alarm. Makes the two detectors' scores comparable."""
    score = row.get(f"{detector}_score")
    threshold = row.get("threshold" if detector == "driftlens" else "mcddd_threshold")
    if score is None or not threshold:
        return None
    return score / threshold


# ============================================================
# Classification metrics
# ============================================================

def calculate_classification_metrics(rows, scenario, detector):
    truth = GROUND_TRUTH[scenario]

    # Only evaluate post-reference windows.
    stream_rows = rows[REFERENCE_WINDOWS:]

    y_true = []
    y_pred = []

    for i, row in enumerate(stream_rows):
        position = REFERENCE_WINDOWS + i

        if truth["concept_drift"]:
            true_drift = (
                truth["drift_start"]
                <= position
                <= len(rows) - 1
            )
        else:
            true_drift = False

        predicted = detector_alarm(row, detector)

        y_true.append(true_drift)
        y_pred.append(predicted)

    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))

    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)

    false_alarm_rate = safe_rate(fp, fp + tn)

    return {
        "scenario": scenario,
        "detector": detector,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "stream_windows": len(stream_rows),
    }


# ============================================================
# Detection delay
# ============================================================

def calculate_detection_delay(rows, scenario, detector):
    truth = GROUND_TRUTH[scenario]

    if not truth["concept_drift"]:
        return None

    start = truth["drift_start"]
    end = truth["drift_end"]

    first_alarm = None

    for position in range(start, len(rows)):
        if detector_alarm(rows[position], detector):
            first_alarm = position
            break

    if first_alarm is None:
        return {
            "first_alarm_position": None,
            "detection_delay": None,
        }

    return {
        "first_alarm_position": first_alarm,
        "detection_delay": first_alarm - start,
    }


# ============================================================
# Alarm agreement
# ============================================================

def calculate_agreement(rows):
    stream_rows = rows[REFERENCE_WINDOWS:]

    both_alarm = 0
    both_no_alarm = 0
    driftlens_only = 0
    mcddd_only = 0

    for row in stream_rows:
        dl = detector_alarm(row, "driftlens")
        mc = detector_alarm(row, "mcddd")

        if dl and mc:
            both_alarm += 1
        elif not dl and not mc:
            both_no_alarm += 1
        elif dl:
            driftlens_only += 1
        else:
            mcddd_only += 1

    total = len(stream_rows)

    return {
        "both_alarm": both_alarm,
        "both_no_alarm": both_no_alarm,
        "driftlens_only": driftlens_only,
        "mcddd_only": mcddd_only,
        "agreement_rate": safe_rate(
            both_alarm + both_no_alarm,
            total
        ),
        "disagreement_rate": safe_rate(
            driftlens_only + mcddd_only,
            total
        ),
    }


# ============================================================
# Score / performance metrics
# ============================================================

def calculate_performance(rows, scenario):
    stream_rows = rows[REFERENCE_WINDOWS:]

    latency = [
        r["processing_latency_ms"]
        for r in stream_rows
        if r.get("processing_latency_ms") is not None
    ]

    throughput = [
        r["throughput_posts_per_sec"]
        for r in stream_rows
        if r.get("throughput_posts_per_sec") is not None
    ]

    return {
        "scenario": scenario,
        "mean_latency_ms": statistics.mean(latency) if latency else None,
        "median_latency_ms": statistics.median(latency) if latency else None,
        "mean_throughput_posts_sec": (
            statistics.mean(throughput) if throughput else None
        ),
        "median_throughput_posts_sec": (
            statistics.median(throughput) if throughput else None
        ),
    }


# ============================================================
# Window-level comparison
# ============================================================

def build_window_comparison(rows, scenario):
    output = []

    truth = GROUND_TRUTH[scenario]

    for i, row in enumerate(rows):
        position = i

        if position < REFERENCE_WINDOWS:
            ground_truth = False
        elif truth["concept_drift"]:
            ground_truth = (
                truth["drift_start"]
                <= position
                <= len(rows) - 1
            )
        else:
            ground_truth = False

        dl_alarm = detector_alarm(row, "driftlens")
        mc_alarm = detector_alarm(row, "mcddd")

        if dl_alarm and mc_alarm:
            agreement = "both_alarm"
        elif not dl_alarm and not mc_alarm:
            agreement = "both_no_alarm"
        elif dl_alarm:
            agreement = "driftlens_only"
        else:
            agreement = "mcddd_only"

        output.append({
            "scenario": scenario,
            "position": position,
            "window_id": row.get("window_id"),
            "window_start": row.get("window_start"),
            "ground_truth_drift": ground_truth,
            "driftlens_score": row.get("driftlens_score"),
            "driftlens_threshold": row.get("threshold"),
            "driftlens_ratio": threshold_ratio(row, "driftlens"),
            "driftlens_alarm": dl_alarm,
            "mcddd_score": row.get("mcddd_score"),
            "mcddd_threshold": row.get("mcddd_threshold"),
            "mcddd_ratio": threshold_ratio(row, "mcddd"),
            "mcddd_alarm": mc_alarm,
            "agreement": agreement,
            "latency_ms": row.get("processing_latency_ms"),
            "throughput_posts_sec": row.get(
                "throughput_posts_per_sec"
            ),
        })

    return output


# ============================================================
# CSV writer
# ============================================================

def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Plots
# ============================================================

def make_plots(all_windows):
    # --------------------------------------------------------
    # 1. Alarm comparison
    # --------------------------------------------------------

    scenarios = []
    dl_counts = []
    mc_counts = []

    for scenario in SCENARIOS:
        rows = [
            r for r in all_windows
            if r["scenario"] == scenario
            and r["position"] >= REFERENCE_WINDOWS
        ]

        scenarios.append(scenario)
        dl_counts.append(
            sum(r["driftlens_alarm"] for r in rows)
        )
        mc_counts.append(
            sum(r["mcddd_alarm"] for r in rows)
        )

    x = list(range(len(scenarios)))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - width / 2 for i in x],
        dl_counts,
        width,
        label="DriftLens"
    )

    plt.bar(
        [i + width / 2 for i in x],
        mc_counts,
        width,
        label="MCD-DD"
    )

    plt.xticks(x, scenarios)
    plt.ylabel("Number of alarms")
    plt.xlabel("Scenario")
    plt.title("Detector Alarm Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "alarm_comparison.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # 2. Score comparison
    # --------------------------------------------------------

    for scenario in SCENARIOS:
        rows = [
            r for r in all_windows
            if r["scenario"] == scenario
            and r["position"] >= REFERENCE_WINDOWS
        ]

        positions = [r["position"] for r in rows]

        # Raw scores have different units (DriftLens ~0.02-0.06, MCD-DD ~0.5-1.5), so plot
        # score / threshold: both detectors alarm above 1.
        dl_scores = [
            r["driftlens_ratio"]
            for r in rows
        ]

        mc_scores = [
            r["mcddd_ratio"]
            for r in rows
        ]

        plt.figure(figsize=(10, 5))

        plt.plot(
            positions,
            dl_scores,
            marker=".",
            label="DriftLens"
        )

        plt.plot(
            positions,
            mc_scores,
            marker=".",
            label="MCD-DD"
        )

        plt.axhline(
            1.0,
            color="grey",
            linewidth=1,
            label="Alarm threshold"
        )

        truth = GROUND_TRUTH[scenario]

        if truth["concept_drift"]:
            plt.axvline(
                truth["drift_start"],
                linestyle="--",
                label="Drift start"
            )

            if truth["drift_end"] != truth["drift_start"]:
                plt.axvline(
                    truth["drift_end"],
                    linestyle=":",
                    label="Drift end"
                )

        elif truth.get("volume_change") is not None:
            plt.axvline(
                truth["volume_change"],
                linestyle="--",
                label="Volume change"
            )

        plt.xlabel("Window position")
        plt.ylabel("Score / threshold (alarm if > 1)")
        plt.title(f"Detector Scores — {scenario}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{scenario}_score_comparison.png",
            dpi=200
        )
        plt.close()

    # --------------------------------------------------------
    # 3. Latency comparison
    # --------------------------------------------------------

    latency_by_scenario = {}

    for scenario in SCENARIOS:
        values = [
            r["latency_ms"]
            for r in all_windows
            if r["scenario"] == scenario
            and r["position"] >= REFERENCE_WINDOWS
            and r["latency_ms"] is not None
        ]

        latency_by_scenario[scenario] = values

    plt.figure(figsize=(9, 5))

    plt.boxplot(
        [
            latency_by_scenario[s]
            for s in SCENARIOS
        ],
        tick_labels=SCENARIOS
    )

    plt.ylabel("Processing latency (ms)")
    plt.xlabel("Scenario")
    plt.title("Processing Latency Comparison")
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "latency_comparison.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # 4. Throughput comparison
    # --------------------------------------------------------

    throughput_by_scenario = {}

    for scenario in SCENARIOS:
        values = [
            r["throughput_posts_sec"]
            for r in all_windows
            if r["scenario"] == scenario
            and r["position"] >= REFERENCE_WINDOWS
            and r["throughput_posts_sec"] is not None
        ]

        throughput_by_scenario[scenario] = values

    plt.figure(figsize=(9, 5))

    plt.boxplot(
        [
            throughput_by_scenario[s]
            for s in SCENARIOS
        ],
        tick_labels=SCENARIOS
    )

    plt.ylabel("Throughput (posts/sec)")
    plt.xlabel("Scenario")
    plt.title("Throughput Comparison")
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "throughput_comparison.png",
        dpi=200
    )

    plt.close()


# ============================================================
# Main evaluation
# ============================================================

def build_comparison_summary(classification_rows, agreement_rows):
    """One row per scenario with both detectors side by side (comparison_summary.csv)."""
    by_key = {(r["scenario"], r["detector"]): r for r in classification_rows}
    agreement = {r["scenario"]: r for r in agreement_rows}
    output = []

    for scenario in SCENARIOS:
        row = {"scenario": scenario}
        for metric in ["precision", "recall", "f1", "false_alarm_rate", "detection_delay"]:
            for detector in DETECTORS:
                row[f"{detector}_{metric}"] = by_key[(scenario, detector)][metric]
        row["agreement_rate"] = agreement[scenario]["agreement_rate"]
        row["disagreement_rate"] = agreement[scenario]["disagreement_rate"]
        output.append(row)

    return output


def main():
    global INPUT_DIR

    parser = argparse.ArgumentParser(description="Person 3 evaluation of DriftLens vs MCD-DD")
    parser.add_argument(
        "--input-dir",
        help="Read DIR/<scenario>/drift_results.jsonl instead of the scenario configs' scores_path"
    )
    args = parser.parse_args()
    if args.input_dir:
        INPUT_DIR = Path(args.input_dir).resolve()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    classification_rows = []
    performance_rows = []
    agreement_rows = []
    window_rows = []

    for scenario in SCENARIOS:

        print(f"\nEvaluating: {scenario}")

        rows = load_results(scenario)

        print(f"  Windows: {len(rows)} ({results_path(scenario).relative_to(PROJECT_ROOT)})")

        # Classification metrics
        for detector in DETECTORS:

            metrics = calculate_classification_metrics(
                rows,
                scenario,
                detector
            )

            delay = calculate_detection_delay(
                rows,
                scenario,
                detector
            )

            if delay is not None:
                metrics.update(delay)
            else:
                metrics["first_alarm_position"] = None
                metrics["detection_delay"] = None

            classification_rows.append(metrics)

            print(
                f"  {detector}: "
                f"Precision={metrics['precision']:.3f}, "
                f"Recall={metrics['recall']:.3f}, "
                f"F1={metrics['f1']:.3f}, "
                f"FAR={metrics['false_alarm_rate']:.3f}, "
                f"Delay={metrics['detection_delay']}"
            )

        # Agreement
        agreement = calculate_agreement(rows)

        agreement["scenario"] = scenario

        agreement_rows.append(agreement)

        # Performance
        performance_rows.append(
            calculate_performance(
                rows,
                scenario
            )
        )

        # Window comparison
        window_rows.extend(
            build_window_comparison(
                rows,
                scenario
            )
        )

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------

    write_csv(
        OUT_DIR / "metrics.csv",
        classification_rows
    )

    write_csv(
        OUT_DIR / "detector_agreement.csv",
        agreement_rows
    )

    write_csv(
        OUT_DIR / "comparison_summary.csv",
        build_comparison_summary(classification_rows, agreement_rows)
    )

    write_csv(
        OUT_DIR / "performance.csv",
        performance_rows
    )

    write_csv(
        OUT_DIR / "window_comparison.csv",
        window_rows
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    make_plots(window_rows)

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "reference_windows": REFERENCE_WINDOWS,
        "scenarios": SCENARIOS,
        "classification_metrics": classification_rows,
        "agreement": agreement_rows,
        "performance": performance_rows,
    }

    with open(
        OUT_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 60)
    print("PERSON 3 EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print("\nGenerated:")
    print("  metrics.csv")
    print("  detector_agreement.csv")
    print("  comparison_summary.csv")
    print("  performance.csv")
    print("  window_comparison.csv")
    print("  evaluation_summary.json")
    print("  plots/")


if __name__ == "__main__":
    main()
