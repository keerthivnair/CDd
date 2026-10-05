"""Compare detector alarms against each scenario's ground truth.

Reads experiments/scenarios/<name>/scenario.json and the detector output written by
    python drift/run_drift.py --config experiments/scenarios/<name>/config.yaml --refit
and writes experiments/results/summary.json + summary.md.

Run from the repository root:
    python experiments/evaluate.py
"""
import argparse
import json
import logging
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = ["natural", "sudden", "gradual", "volume", "no_drift"]
DETECTORS = {"driftlens": "driftlens_alarm", "mcddd": "mcddd_alarm"}
log = logging.getLogger("experiments.evaluate")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def rate(flags: list):
    return round(sum(flags) / len(flags), 4) if flags else None


def evaluate(truth: dict, scores: list, n_ref: int, tolerance: int) -> dict:
    """Metrics for one scenario. scores are run_drift.py rows in stream order (position = index)."""
    start = truth["drift_start"]["position"] if truth.get("drift_start") else None
    end = truth["drift_end"]["position"] if truth.get("drift_end") else None
    vol = truth["volume_change"]["position"] if truth.get("volume_change") else None
    out = {}
    for det, key in DETECTORS.items():
        alarms = [bool(r.get(key)) for r in scores]
        stream = list(range(n_ref, len(alarms)))
        m = {"alarms": sum(alarms[p] for p in stream), "stream_windows": len(stream),
             "alarm_rate": rate([alarms[p] for p in stream]),
             "alarm_positions": [p for p in stream if alarms[p]]}
        if start is not None:
            before = [alarms[p] for p in stream if p < start]
            first = next((p for p in range(start, len(alarms)) if alarms[p]), None)
            m.update({
                "false_alarms_before_drift": sum(before),
                "false_alarm_rate_before_drift": rate(before),
                "first_alarm_after_drift": first,
                "detection_delay": None if first is None else first - start,
                "detected_within_tolerance": first is not None and first - start <= tolerance + (end - start),
                "alarm_rate_after_drift": rate(alarms[start:]),
            })
        elif vol is not None:
            m.update({"false_alarm_rate_before_volume_change": rate([alarms[p] for p in stream if p < vol]),
                      "false_alarm_rate_after_volume_change": rate(alarms[vol:]),
                      "alarm_at_volume_change": alarms[vol]})
        elif truth.get("concept_drift") is False:
            m["false_alarm_rate"] = m["alarm_rate"]
        out[det] = m
    return out


def to_markdown(summary: dict) -> str:
    lines = ["# Scenario evaluation summary", "",
             f"Stream windows = windows after the {summary['reference_windows']} reference windows. "
             f"Detection counts as on time if the first alarm comes within {summary['tolerance']} windows "
             "of the drift end (gradual) or drift start (sudden).", "",
             "| Scenario | Detector | Expected | Stream alarms | False alarms before change | "
             "First alarm (delay) | Alarm rate after change |",
             "|---|---|---|---|---|---|---|"]
    for name, res in summary["scenarios"].items():
        if "missing" in res:
            lines.append(f"| {name} | - | - | not run | - | - | - |")
            continue
        t = res["ground_truth"]
        if t.get("drift_start"):
            exp = f"drift at {t['drift_start']['position']}" + (
                f"-{t['drift_end']['position']}" if t["drift_end"]["position"] != t["drift_start"]["position"] else "")
        elif t.get("volume_change"):
            exp = f"no drift (volume x{t['volume_change']['factor']} at {t['volume_change']['position']})"
        elif t.get("concept_drift") is False:
            exp = "no drift"
        else:
            exp = "unknown (real)"
        for det, m in res["metrics"].items():
            if "first_alarm_after_drift" in m:
                fa = f"{m['false_alarms_before_drift']} ({m['false_alarm_rate_before_drift']:.0%})"
                first = "none" if m["first_alarm_after_drift"] is None else \
                    f"{m['first_alarm_after_drift']} (+{m['detection_delay']})"
                after = f"{m['alarm_rate_after_drift']:.0%}"
            elif "false_alarm_rate_before_volume_change" in m:
                fa = f"{m['false_alarm_rate_before_volume_change']:.0%}"
                first = "-"
                after = f"{m['false_alarm_rate_after_volume_change']:.0%} (any alarm is false)"
            else:
                fa, first, after = ("-", "-", "-")
            lines.append(f"| {name} | {det} | {exp} | {m['alarms']}/{m['stream_windows']} | {fa} | {first} | {after} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector alarms against scenario ground truth")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--tolerance", type=int, default=3, help="Allowed delay (windows) after the drift end")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    n_ref = cfg["stream"]["reference_windows"]
    summary = {"reference_windows": n_ref, "tolerance": args.tolerance, "scenarios": {}}
    for name in SCENARIOS:
        sdir = resolve(os.path.join(cfg["scenarios_dir"], name))
        with open(os.path.join(sdir, "scenario.json")) as f:
            scen = json.load(f)
        with open(os.path.join(sdir, "config.yaml")) as f:
            scores_path = resolve(yaml.safe_load(f)["output"]["scores_path"])
        if not os.path.exists(scores_path):
            log.warning("No detector output for '%s' at %s; run drift/run_drift.py first", name, scores_path)
            summary["scenarios"][name] = {"missing": scores_path}
            continue
        with open(scores_path) as f:
            scores = [json.loads(line) for line in f if line.strip()]
        if len(scores) != scen["n_windows"]:
            log.warning("'%s': %s scored windows, scenario has %s", name, len(scores), scen["n_windows"])
        summary["scenarios"][name] = {"ground_truth": scen["ground_truth"],
                                      "metrics": evaluate(scen["ground_truth"], scores, n_ref, args.tolerance)}

    out_dir = resolve("experiments/results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    md = to_markdown(summary)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
