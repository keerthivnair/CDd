"""
Unit tests for Task 4: InfluxDB + Grafana monitoring pipeline.
Covers schema validation against real Task 3 output, Line Protocol conversion,
deduplication, and CSV backup logging. Uses Python standard library unittest.
"""

import os
import sys
import csv
import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure root directory is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.schema import WindowDriftResult
from monitoring.client import DriftMetricsLogger


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

    def test_line_protocol_formatting(self):
        """Verify that to_line_protocol formats correct InfluxDB line protocol."""
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
            "throughput_posts_per_sec": 616.2,
            "cpu_pct": 25.0,
            "mem_mb": 1024.0
        }
        res = WindowDriftResult.from_task3_dict(task3_sample)
        lp = res.to_line_protocol("drift_metrics")
        self.assertTrue(lp.startswith("drift_metrics,pipeline=cdd-stream,window_id=18402,driftlens_alarm_state=drift,is_reference=false"))
        self.assertIn("driftlens_score=0.029159", lp)
        self.assertIn("driftlens_alarm=1i", lp)
        self.assertIn("driftlens_threshold=0.028602", lp)
        self.assertIn("processing_latency_ms=234.5", lp)
        self.assertIn("throughput_posts_per_sec=616.2", lp)
        self.assertIn("cpu_pct=25.0", lp)
        self.assertIn("mem_mb=1024.0", lp)
        self.assertIn("post_count=1445i", lp)
        self.assertIn("n_used=1000i", lp)

    def test_validation_missing_fields(self):
        """Ensure invalid Task 3 payloads raise ValueError instead of inserting corrupt data."""
        # Missing window_id
        with self.assertRaises(ValueError):
            WindowDriftResult.from_task3_dict({"driftlens_score": 0.05, "driftlens_alarm": False})

        # Missing driftlens_score
        with self.assertRaises(ValueError):
            WindowDriftResult.from_task3_dict({"window_id": 1, "driftlens_alarm": False})

        # Missing driftlens_alarm
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

            # First insert succeeds
            first = logger.log_drift_result(task3_sample)
            self.assertTrue(first)

            # Second insert of identical window_id is rejected by deduplication
            second = logger.log_drift_result(task3_sample)
            self.assertFalse(second)

            logger.close()

            # Check CSV has only 1 row
            with open(csv_file, "r") as f:
                rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["window_id"], "18371")


if __name__ == "__main__":
    unittest.main()
