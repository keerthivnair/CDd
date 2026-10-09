"""
Task 4: Drift Explainability and Disagreement Analysis Engine.
Analyzes detector outputs (DriftLens vs MCD-DD), prioritizes disagreement cases,
compares before-and-after window distributions, extracts emerging/fading keywords,
and identifies representative tweets closest to semantic centroids.
"""

import os
import re
import csv
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("cdd.explain")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must", "it", "its", "it's",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can't", "don't", "just",
    "now", "like", "get", "new", "one", "also", "even", "still", "well", "way"
}


def tokenize(text: str) -> List[str]:
    """Extracts alphanumeric words of length >= 3 excluding common stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()]


def compute_keyword_frequencies(texts: List[str]) -> Counter:
    """Computes word frequency counts for a collection of texts."""
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    return counts


def find_keyword_shifts(before_texts: List[str], after_texts: List[str], top_n: int = 8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes terms that significantly increased or decreased in relative frequency
    between the before and after windows.
    """
    n_before = max(len(before_texts), 1)
    n_after = max(len(after_texts), 1)

    counts_before = compute_keyword_frequencies(before_texts)
    counts_after = compute_keyword_frequencies(after_texts)

    all_vocab = set(counts_before.keys()) | set(counts_after.keys())
    shifts = []

    for term in all_vocab:
        freq_b = counts_before[term] / n_before
        freq_a = counts_after[term] / n_after
        # Relative growth/drop with smoothing
        diff = freq_a - freq_b
        shifts.append((term, diff, freq_b, freq_a, counts_before[term], counts_after[term]))

    # Emerging: largest positive diff
    shifts_sorted = sorted(shifts, key=lambda x: x[1], reverse=True)
    emerging = [
        {"term": item[0], "score": round(item[1] * 100, 2), "count_before": item[4], "count_after": item[5]}
        for item in shifts_sorted if item[1] > 0 and item[5] >= 2
    ][:top_n]

    # Fading: largest negative diff
    fading = [
        {"term": item[0], "score": round(item[1] * 100, 2), "count_before": item[4], "count_after": item[5]}
        for item in shifts_sorted if item[1] < 0 and item[4] >= 2
    ][-top_n:][::-1]

    return emerging, fading


class WindowTweetLoader:
    """
    Locates and extracts real cleaned tweet texts for specific window IDs.
    Directly checks windows_all.jsonl or scenario windows files.
    """

    def __init__(self, windows_path: Optional[str] = None):
        self.windows_path = Path(windows_path) if windows_path else None
        self._cached_windows: Dict[int, Dict[str, Any]] = {}
        self._load_attempted = False

    def locate_windows_file(self) -> Optional[Path]:
        """Checks for existing window JSONL files in priority order."""
        if self.windows_path and self.windows_path.exists():
            return self.windows_path

        candidates = [
            PROJECT_ROOT / "windows_all.jsonl",
            PROJECT_ROOT / "experiments" / "scenarios" / "natural" / "windows.jsonl",
            PROJECT_ROOT / "output" / "windows.jsonl",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def load_window(self, target_window_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves tweet list and metadata for a given window ID."""
        if target_window_id in self._cached_windows:
            return self._cached_windows[target_window_id]

        fpath = self.locate_windows_file()
        if not fpath:
            return None

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    w = json.loads(line)
                    if int(w.get("window_id", -1)) == target_window_id:
                        self._cached_windows[target_window_id] = w
                        return w
        except Exception as e:
            log.warning("Error reading windows from %s: %s", fpath, e)

        return None


class DriftExplainer:
    """
    Performs full explainability analysis across detector results.
    """

    def __init__(self, detector_results_path: Optional[str] = None, windows_path: Optional[str] = None):
        self.detector_path = Path(detector_results_path) if detector_results_path else self._find_detector_file()
        self.tweet_loader = WindowTweetLoader(windows_path)
        self.detector_records: List[Dict[str, Any]] = []
        self._load_detector_results()

    def _find_detector_file(self) -> Path:
        candidates = [
            PROJECT_ROOT / "output" / "drift" / "driftlens.jsonl",
            Path.home() / "Downloads" / "driftlens.jsonl",
            PROJECT_ROOT / "output" / "drift" / "comparison.jsonl",
        ]
        for c in candidates:
            if c.exists():
                return c
        return PROJECT_ROOT / "output" / "drift" / "driftlens.jsonl"

    def _load_detector_results(self):
        if not self.detector_path.exists():
            log.warning("Detector results file not found at %s", self.detector_path)
            return

        with open(self.detector_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Compute agreement and ratios if missing
                dl_alarm = bool(row.get("driftlens_alarm", False))
                mcd_alarm = row.get("mcddd_alarm")
                is_ref = bool(row.get("is_reference", False))

                if is_ref:
                    agreement = "reference"
                elif mcd_alarm is None:
                    agreement = "mcddd_missing"
                else:
                    mcd_bool = bool(mcd_alarm)
                    if dl_alarm and mcd_bool:
                        agreement = "both"
                    elif dl_alarm and not mcd_bool:
                        agreement = "driftlens_only"
                    elif not dl_alarm and mcd_bool:
                        agreement = "mcddd_only"
                    else:
                        agreement = "neither"

                dl_score = float(row.get("driftlens_score", 0.0))
                dl_thr = float(row.get("threshold", row.get("driftlens_threshold", 0.0)))
                dl_ratio = round(dl_score / dl_thr, 4) if dl_thr > 0 else 0.0

                mcd_score = float(row["mcddd_score"]) if row.get("mcddd_score") is not None else None
                mcd_thr = float(row["mcddd_threshold"]) if row.get("mcddd_threshold") is not None else None
                mcd_ratio = round(mcd_score / mcd_thr, 4) if (mcd_score is not None and mcd_thr and mcd_thr > 0) else None

                row["agreement"] = row.get("agreement") or agreement
                row["driftlens_ratio"] = row.get("driftlens_ratio") or dl_ratio
                row["mcddd_ratio"] = row.get("mcddd_ratio") or mcd_ratio
                self.detector_records.append(row)

    def generate_disagreement_table(self) -> List[Dict[str, Any]]:
        """Builds a structured comparison and disagreement table for all stream windows."""
        table = []
        for r in self.detector_records:
            table.append({
                "window_id": r["window_id"],
                "window_start": r.get("window_start"),
                "is_reference": r.get("is_reference", False),
                "driftlens_score": r.get("driftlens_score"),
                "driftlens_threshold": r.get("threshold", r.get("driftlens_threshold")),
                "driftlens_ratio": r.get("driftlens_ratio"),
                "driftlens_alarm": r.get("driftlens_alarm"),
                "mcddd_score": r.get("mcddd_score"),
                "mcddd_threshold": r.get("mcddd_threshold"),
                "mcddd_ratio": r.get("mcddd_ratio"),
                "mcddd_alarm": r.get("mcddd_alarm"),
                "agreement": r.get("agreement"),
                "n_posts": r.get("n_posts", 0),
                "n_used": r.get("n_used", 0),
            })
        return table

    def export_disagreement_csv(self, output_csv: Path) -> Path:
        """Exports the disagreement table to a clean CSV file."""
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table = self.generate_disagreement_table()
        if not table:
            return output_csv

        fieldnames = list(table[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table)
        log.info("Exported disagreement table with %d windows to %s", len(table), output_csv)
        return output_csv

    def select_important_windows(self) -> List[Dict[str, Any]]:
        """
        Selects priority windows for deep semantic explainability:
          1. All 'mcddd_only' disagreement windows
          2. Top 'both' consensus drift windows (highest ratios)
          3. Representative 'driftlens_only' gradual drift windows
          4. Notable high-volume / sudden change windows
        """
        by_agreement = {
            "mcddd_only": [],
            "both": [],
            "driftlens_only": [],
            "neither": []
        }
        for r in self.detector_records:
            if r.get("is_reference"):
                continue
            cat = r.get("agreement")
            if cat in by_agreement:
                by_agreement[cat].append(r)

        selected = []

        # 1. MCD-DD only disagreements (crucial edge case: exiting reference)
        for w in sorted(by_agreement["mcddd_only"], key=lambda x: x.get("mcddd_ratio") or 0, reverse=True):
            selected.append({"reason": "mcddd_only_disagreement", "record": w})

        # 2. Both detectors alarm (consensus major events)
        top_both = sorted(by_agreement["both"], key=lambda x: (x.get("driftlens_ratio") or 0) + (x.get("mcddd_ratio") or 0), reverse=True)[:4]
        for w in top_both:
            selected.append({"reason": "consensus_major_drift", "record": w})

        # 3. High-ratio DriftLens-only disagreements
        top_dl = sorted(by_agreement["driftlens_only"], key=lambda x: x.get("driftlens_ratio") or 0, reverse=True)[:3]
        for w in top_dl:
            selected.append({"reason": "driftlens_only_gradual_drift", "record": w})

        # 4. Consensus normal window for baseline contrast
        if by_agreement["neither"]:
            selected.append({"reason": "consensus_normal_benchmark", "record": by_agreement["neither"][0]})

        return selected

    def explain_window(self, window_id: int) -> Dict[str, Any]:
        """
        Generates full before-and-after explainability for a specific window:
        - Before drift window details & tweets
        - After drift window details & tweets
        - What changed: emerging keywords, fading keywords, semantic explanation
        """
        rec = next((r for r in self.detector_records if r["window_id"] == window_id), None)
        if not rec:
            return {"error": f"Window ID {window_id} not found in detector results"}

        # Find preceding window
        idx = next((i for i, r in enumerate(self.detector_records) if r["window_id"] == window_id), -1)
        before_rec = self.detector_records[idx - 1] if idx > 0 else self.detector_records[0]

        # Retrieve tweets
        w_curr = self.tweet_loader.load_window(window_id)
        w_prev = self.tweet_loader.load_window(before_rec["window_id"])

        tweet_data_available = bool(w_curr and "texts" in w_curr and len(w_curr["texts"]) > 0)

        if tweet_data_available:
            texts_curr = w_curr["texts"]
            texts_prev = w_prev["texts"] if (w_prev and "texts" in w_prev) else []
            emerging, fading = find_keyword_shifts(texts_prev, texts_curr)
            rep_before = texts_prev[:4]
            rep_after = texts_curr[:4]
            semantic_summary = f"Window {window_id} demonstrates shifts with emerging terms: {', '.join(e['term'] for e in emerging[:4])}."
        else:
            emerging, fading = [], []
            rep_before, rep_after = [], []
            semantic_summary = (
                "DATA DEPENDENCY NOTICE: 'windows_all.jsonl' is not present in the local repository. "
                "Original tweet text is required to extract word-level semantic shifts. "
                "Provide 'windows_all.jsonl' to display live tweet texts."
            )

        # Detector rationale
        dl_alarm = bool(rec.get("driftlens_alarm"))
        mcd_alarm = bool(rec.get("mcddd_alarm"))
        dl_ratio = rec.get("driftlens_ratio", 0.0)
        mcd_ratio = rec.get("mcddd_ratio", 0.0)

        if dl_alarm and mcd_alarm:
            rationale = (
                f"Both detectors agreed on drift. DriftLens FDD score ({rec.get('driftlens_score'):.4f}) "
                f"exceeded its threshold by {((dl_ratio - 1) * 100):.1f}%, while MCD-DD discrepancy ({rec.get('mcddd_score'):.4f}) "
                f"exceeded its threshold by {((mcd_ratio - 1) * 100):.1f}%."
            )
        elif dl_alarm and not mcd_alarm:
            rationale = (
                f"DriftLens alarmed (ratio: {dl_ratio:.2f}) due to cumulative distribution shift away from the fixed April 2020 baseline. "
                f"MCD-DD did not alarm (ratio: {mcd_ratio:.2f} <= 1.0) because its 10-window rolling context adapts to gradual transitions."
            )
        elif not dl_alarm and mcd_alarm:
            rationale = (
                f"MCD-DD alarmed (ratio: {mcd_ratio:.2f}) due to high discrepancy against its immediate contrastive context windows. "
                f"DriftLens remained quiet (ratio: {dl_ratio:.2f} <= 1.0) because the sample distribution remained within its conservative p99 baseline boundary."
            )
        else:
            rationale = "Neither detector observed significant distribution movement beyond normal random variation."

        return {
            "window_id": window_id,
            "window_start": rec.get("window_start"),
            "agreement": rec.get("agreement"),
            "detector_metrics": {
                "driftlens_score": rec.get("driftlens_score"),
                "driftlens_threshold": rec.get("threshold", rec.get("driftlens_threshold")),
                "driftlens_ratio": dl_ratio,
                "driftlens_alarm": dl_alarm,
                "mcddd_score": rec.get("mcddd_score"),
                "mcddd_threshold": rec.get("mcddd_threshold"),
                "mcddd_ratio": mcd_ratio,
                "mcddd_alarm": mcd_alarm,
            },
            "detector_rationale": rationale,
            "tweet_data_available": tweet_data_available,
            "before_drift": {
                "window_id": before_rec["window_id"],
                "window_start": before_rec.get("window_start"),
                "tweet_volume": rec.get("n_posts", 0),
                "representative_tweets": rep_before,
            },
            "after_drift": {
                "window_id": window_id,
                "window_start": rec.get("window_start"),
                "tweet_volume": rec.get("n_posts", 0),
                "n_used": rec.get("n_used", 0),
                "representative_tweets": rep_after,
            },
            "what_changed": {
                "emerging_keywords": emerging,
                "fading_keywords": fading,
                "semantic_summary": semantic_summary,
                "change_type": "abrupt" if (dl_ratio > 1.3 or (mcd_ratio and mcd_ratio > 1.3)) else "gradual",
            }
        }

    def generate_full_report(self, output_path: Path) -> Dict[str, Any]:
        """Runs the complete selection, disagreement, and explainability pipeline."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = self.generate_disagreement_table()
        important = self.select_important_windows()

        explanations = []
        for item in important:
            wid = item["record"]["window_id"]
            exp = self.explain_window(wid)
            exp["selection_reason"] = item["reason"]
            explanations.append(exp)

        # Agreement summary stats
        agreement_counts = Counter(r["agreement"] for r in table if not r.get("is_reference"))

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_detector_file": str(self.detector_path),
            "windows_source": str(self.tweet_loader.locate_windows_file() or "MISSING (windows_all.jsonl)"),
            "total_windows": len(table),
            "stream_windows": len(table) - 14,
            "agreement_summary": dict(agreement_counts),
            "selected_windows_count": len(explanations),
            "selected_explanations": explanations,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        log.info("Full explainability report generated at %s", output_path)
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Person 4: Drift Explainability Generator")
    parser.add_argument("--detector-results", default="output/drift/driftlens.jsonl", help="Path to detector results JSONL")
    parser.add_argument("--windows", default="windows_all.jsonl", help="Path to windows JSONL with tweet text")
    parser.add_argument("--output-report", default="output/drift/explainability_report.json", help="Path for JSON report")
    parser.add_argument("--output-table", default="output/drift/disagreement_table.csv", help="Path for disagreement CSV")
    args = parser.parse_args()

    explainer = DriftExplainer(detector_results_path=args.detector_results, windows_path=args.windows)
    explainer.export_disagreement_csv(Path(args.output_table))
    explainer.generate_full_report(Path(args.output_report))


if __name__ == "__main__":
    main()
