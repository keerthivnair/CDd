"""
Comprehensive Unit & Integration tests for Person 4: Drift Explainability + Grafana Monitoring.
Covers:
  - Real detector dictionary parsing and schema validation
  - Agreement classification logic across all 5 states
  - MCD-DD metric handling, threshold ratio normalization, and Line Protocol serialization
  - InfluxDB logger deduplication and CSV export
  - TF-IDF keyword shift analysis (emerging & fading terms)
  - Priority window selection for explainability
  - Honest handling of missing raw tweet storage without hallucinations
"""

import os
import sys
import csv
import json
import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure root directory is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.schema import WindowDriftResult, classify_agreement, AGREEMENT_CODES
from monitoring.client import DriftMetricsLogger
from monitoring.explain import find_keyword_shifts, DriftExplainer, WindowTweetLoader


class TestTask4Monitoring(unittest.TestCase):

    def test_from_real_task3_dict(self):
        """Verify that WindowDriftResult correctly parses the real Task 3 dictionary."""
        task3_sample = {
            "window_id": 18402,
            "window_start": "2020-05-20T00:00:00",
            "driftlens_score": 0.029159,
            "driftlens_alarm": True,
            "threshold": 0.028602,
            "n_posts": 1445,
            "n_used": 1000,
            "is_reference": False,
            "processing_latency_ms": 234.5,
            "throughput_posts_per_sec": 616.2
        }

        res = WindowDriftResult.from_task3_dict(task3_sample)
        self.assertEqual(res.window_id, 18402)
        self.assertEqual(res.driftlens_score, 0.029159)
        self.assertTrue(res.driftlens_alarm)
        self.assertEqual(res.driftlens_threshold, 0.028602)
        self.assertEqual(res.post_count, 1445)
        self.assertEqual(res.n_used, 1000)
        self.assertFalse(res.is_reference)
        self.assertEqual(res.processing_latency_ms, 234.5)
        self.assertEqual(res.throughput_posts_per_sec, 616.2)

        dt = res.get_datetime()
        self.assertEqual(dt.year, 2020)
        self.assertEqual(dt.month, 5)
        self.assertEqual(dt.day, 20)

    def test_agreement_classification(self):
        """Test classify_agreement across all conditions."""
        # 1. Reference period
        self.assertEqual(classify_agreement(is_reference=True, dl_alarm=True, mcd_alarm=True), "reference")
        self.assertEqual(classify_agreement(is_reference=True, dl_alarm=False, mcd_alarm=False), "reference")

        # 2. Both alarmed
        self.assertEqual(classify_agreement(is_reference=False, dl_alarm=True, mcd_alarm=True), "both")

        # 3. DriftLens only alarmed
        self.assertEqual(classify_agreement(is_reference=False, dl_alarm=True, mcd_alarm=False), "driftlens_only")

        # 4. MCD-DD only alarmed (early May 2020 disagreement)
        self.assertEqual(classify_agreement(is_reference=False, dl_alarm=False, mcd_alarm=True), "mcddd_only")

        # 5. Neither alarmed
        self.assertEqual(classify_agreement(is_reference=False, dl_alarm=False, mcd_alarm=False), "neither")

        # 6. MCD-DD missing
        self.assertEqual(classify_agreement(is_reference=False, dl_alarm=True, mcd_alarm=None), "mcddd_missing")

    def test_mcddd_and_ratio_fields(self):
        """Verify MCD-DD scores and normalized score/threshold ratios are accurately computed."""
        record = {
            "window_id": 18804,
            "window_start": "2021-06-25T18:30:00",
            "driftlens_score": 0.069639,
            "driftlens_alarm": True,
            "threshold": 0.029583,
            "mcddd_score": 1.47867,
            "mcddd_alarm": True,
            "mcddd_threshold": 1.1741,
            "n_posts": 1820,
            "n_used": 1000,
            "is_reference": False,
            "processing_latency_ms": 312.0,
            "throughput_posts_per_sec": 583.3
        }
        res = WindowDriftResult.from_task3_dict(record)
        self.assertEqual(res.agreement, "both")
        self.assertEqual(res.agreement_code, 4)
        self.assertEqual(res.mcddd_score, 1.47867)
        self.assertEqual(res.mcddd_threshold, 1.1741)
        self.assertTrue(res.mcddd_alarm)
        # Ratio dl: 0.069639 / 0.029583 ≈ 2.354
        self.assertAlmostEqual(res.driftlens_ratio, 2.354, places=2)
        # Ratio mcd: 1.47867 / 1.1741 ≈ 1.2594
        self.assertAlmostEqual(res.mcddd_ratio, 1.2594, places=2)

    def test_line_protocol_formatting_with_mcddd(self):
        """Verify that to_line_protocol includes agreement tags and MCD-DD fields."""
        record = {
            "window_id": 18387,
            "window_start": "2020-05-04T18:30:00",
            "driftlens_score": 0.024158,
            "driftlens_alarm": False,
            "threshold": 0.029583,
            "mcddd_score": 2.03173,
            "mcddd_alarm": True,
            "mcddd_threshold": 1.1741,
            "n_posts": 1000,
            "n_used": 1000,
            "is_reference": False,
            "processing_latency_ms": 200.0,
            "throughput_posts_per_sec": 500.0
        }
        res = WindowDriftResult.from_task3_dict(record)
        lp = res.to_line_protocol("drift_metrics")
        self.assertIn("agreement=mcddd_only", lp)
        self.assertIn("driftlens_alarm_state=normal", lp)
        self.assertIn("mcddd_alarm_state=drift", lp)
        self.assertIn("driftlens_score=0.024158", lp)
        self.assertIn("driftlens_alarm=0i", lp)
        self.assertIn("mcddd_score=2.03173", lp)
        self.assertIn("mcddd_alarm=1i", lp)
        self.assertIn("agreement_code=3i", lp)

    def test_validation_missing_fields(self):
        """Ensure invalid Task 3 payloads raise ValueError instead of inserting corrupt data."""
        with self.assertRaises(ValueError):
            WindowDriftResult.from_task3_dict({"driftlens_score": 0.05, "driftlens_alarm": False})

        with self.assertRaises(ValueError):
            WindowDriftResult.from_task3_dict({"window_id": 1, "driftlens_alarm": False})

        with self.assertRaises(ValueError):
            WindowDriftResult.from_task3_dict({"window_id": 1, "driftlens_score": 0.05, "window_start": "2020-04-19"})

    def test_deduplication(self):
        """Verify that DriftMetricsLogger avoids inserting duplicate window IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test_dedup.csv"
            cfg = {
                "influxdb": {"url": "http://localhost:9999", "timeout_ms": 500},
                "logging": {"csv_path": str(csv_file), "log_to_csv": True, "log_to_influx": False},
                "system_metrics": {"collect_cpu_mem": False}
            }
            logger = DriftMetricsLogger(config=cfg)

            task3_sample = {
                "window_id": 18371,
                "window_start": "2020-04-19T00:00:00",
                "driftlens_score": 0.025,
                "driftlens_alarm": False,
                "threshold": 0.0286,
                "n_posts": 500,
                "n_used": 500,
                "processing_latency_ms": 100.0,
                "throughput_posts_per_sec": 5000.0
            }

            first = logger.log_drift_result(task3_sample)
            self.assertTrue(first)

            second = logger.log_drift_result(task3_sample)
            self.assertFalse(second)

            logger.close()

            with open(csv_file, "r") as f:
                rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["window_id"], "18371")

    def test_keyword_shifts_computation(self):
        """Test emerging and fading keyword extraction between two document sets."""
        before_corpus = [
            "stay home lockdown quarantine safe mask health",
            "lockdown measures stay home covid quarantine safe",
            "stay safe sanitize hands social distance quarantine",
            "coronavirus update stay inside quarantine home safe"
        ]
        after_corpus = [
            "covid vaccine dose pfizer approved injection second",
            "moderna vaccine appointment dose rollout clinical",
            "pfizer vaccine appointment shot immunity vaccinated",
            "vaccination dose vaccine rollout healthcare hospital"
        ]

        emerging, fading = find_keyword_shifts(before_corpus, after_corpus, top_n=5)
        emerging_terms = [e["term"] for e in emerging]
        fading_terms = [f["term"] for f in fading]

        # "vaccine" and "dose" should be emerging
        self.assertIn("vaccine", emerging_terms)
        self.assertIn("dose", emerging_terms)

        # "quarantine", "stay", "lockdown" should be fading
        self.assertTrue(any(t in fading_terms for t in ["quarantine", "lockdown", "stay"]))

    def test_select_important_windows(self):
        """Verify that select_important_windows isolates disagreement and consensus cases."""
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as tf:
            records = [
                {"window_id": 1, "is_reference": True, "driftlens_alarm": False, "threshold": 1.0, "driftlens_score": 0.5},
                {"window_id": 2, "is_reference": False, "driftlens_alarm": False, "mcddd_alarm": True, "threshold": 1.0, "driftlens_score": 0.5, "mcddd_threshold": 1.0, "mcddd_score": 2.0},
                {"window_id": 3, "is_reference": False, "driftlens_alarm": True, "mcddd_alarm": True, "threshold": 1.0, "driftlens_score": 2.5, "mcddd_threshold": 1.0, "mcddd_score": 2.5},
                {"window_id": 4, "is_reference": False, "driftlens_alarm": True, "mcddd_alarm": False, "threshold": 1.0, "driftlens_score": 1.8, "mcddd_threshold": 1.0, "mcddd_score": 0.3},
                {"window_id": 5, "is_reference": False, "driftlens_alarm": False, "mcddd_alarm": False, "threshold": 1.0, "driftlens_score": 0.2, "mcddd_threshold": 1.0, "mcddd_score": 0.2},
            ]
            for r in records:
                tf.write(json.dumps(r) + "\n")
            tf.flush()

            explainer = DriftExplainer(detector_results_path=tf.name)
            selected = explainer.select_important_windows()

            reasons = [s["reason"] for s in selected]
            self.assertIn("mcddd_only_disagreement", reasons)
            self.assertIn("consensus_major_drift", reasons)
            self.assertIn("driftlens_only_gradual_drift", reasons)
            self.assertIn("consensus_normal_benchmark", reasons)

    def test_missing_tweet_store_honesty(self):
        """Verify that missing raw tweet files are handled safely and reported honestly."""
        loader = WindowTweetLoader(windows_path="/non/existent/path/windows.jsonl")
        window = loader.load_window(99999)
        self.assertIsNone(window)


if __name__ == "__main__":
    unittest.main()
