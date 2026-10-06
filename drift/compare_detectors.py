"""Side-by-side DriftLens vs MCD-DD comparison of a run_drift.py scores file.

Both detectors score the same embeddings in the same run, so each window is one row. The raw
scores have different units (Frechet distance vs. concept-embedding L2 distance), so each detector
also gets score / threshold: > 1 means alarm, and the two ratios can be compared directly.

Run from the repository root:
    python drift/compare_detectors.py                                    # main run (output/drift)
    python drift/compare_detectors.py --scores output/experiments/sudden/scores.jsonl
Writes <scores dir>/comparison.jsonl (one row per window) and comparison_summary.json.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def ratio(score, threshold):
    return None if score is None or not threshold else round(score / threshold, 4)


def compare_row(r: dict) -> dict:
    dl, mcd = bool(r["driftlens_alarm"]), r.get("mcddd_alarm")
    if r.get("is_reference"):
        agreement = "reference"  # MCD-DD does not score its own training windows
    elif mcd is None:
        agreement = "mcddd_missing"
    else:
        agreement = {(True, True): "both", (True, False): "driftlens_only",
                     (False, True): "mcddd_only", (False, False): "neither"}[(dl, bool(mcd))]
    return {
        "window_id": r["window_id"],
        "window_start": r.get("window_start"),
        "n_posts": r.get("n_posts"),
        "is_reference": r.get("is_reference", False),
        "driftlens_score": r["driftlens_score"],
        "driftlens_threshold": r["threshold"],
        "driftlens_ratio": ratio(r["driftlens_score"], r["threshold"]),
        "driftlens_alarm": dl,
        "mcddd_score": r.get("mcddd_score"),
        "mcddd_threshold": r.get("mcddd_threshold"),
        "mcddd_ratio": ratio(r.get("mcddd_score"), r.get("mcddd_threshold")),
        "mcddd_alarm": mcd,
        "mcddd_lag": r.get("mcddd_lag"),
        "agreement": agreement,
    }


def summarize(rows: list) -> dict:
    stream = [r for r in rows if not r["is_reference"]]
    agree = Counter(r["agreement"] for r in stream)
    scored = [r for r in stream if r["mcddd_alarm"] is not None]
    by_month = defaultdict(lambda: {"windows": 0, "driftlens_alarms": 0, "mcddd_alarms": 0})
    for r in stream:
        m = by_month[(r["window_start"] or "")[:7]]
        m["windows"] += 1
        m["driftlens_alarms"] += r["driftlens_alarm"]
        m["mcddd_alarms"] += bool(r["mcddd_alarm"])
    return {
        "stream_windows": len(stream),
        "driftlens_alarms": sum(r["driftlens_alarm"] for r in stream),
        "mcddd_alarms": sum(bool(r["mcddd_alarm"]) for r in stream),
        "agreement": dict(agree),
        "agreement_rate": round((agree["both"] + agree["neither"]) / len(scored), 4) if scored else None,
        "mcddd_alarms_also_flagged_by_driftlens": f"{agree['both']}/{agree['both'] + agree['mcddd_only']}",
        "by_month": dict(sorted(by_month.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare DriftLens and MCD-DD window by window")
    parser.add_argument("--scores", default="output/drift/driftlens.jsonl", help="run_drift.py output")
    parser.add_argument("--out-dir", help="Where to write comparison files (default: next to --scores)")
    args = parser.parse_args()

    src = resolve(args.scores)
    out_dir = resolve(args.out_dir) if args.out_dir else os.path.dirname(src)
    with open(src, encoding="utf-8") as f:
        rows = [compare_row(json.loads(line)) for line in f if line.strip()]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"source": os.path.relpath(src, PROJECT_ROOT), **summarize(rows)}
    with open(os.path.join(out_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"{summary['stream_windows']} stream windows from {summary['source']}")
    print(f"DriftLens alarms: {summary['driftlens_alarms']}   MCD-DD alarms: {summary['mcddd_alarms']}")
    print(f"Agreement: {summary['agreement']}  (agree on {summary['agreement_rate']:.0%} of windows)")
    print(f"{'month':<8} {'windows':>7} {'DriftLens':>9} {'MCD-DD':>7}")
    for month, m in summary["by_month"].items():
        print(f"{month:<8} {m['windows']:>7} {m['driftlens_alarms']:>9} {m['mcddd_alarms']:>7}")
    print(f"-> {os.path.join(out_dir, 'comparison.jsonl')}")


if __name__ == "__main__":
    main()
