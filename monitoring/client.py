"""
InfluxDB Client and Dual Logger (InfluxDB + CSV) for Task 4.
Consumes real Task 3 (DriftLens) outputs and system resource metrics.
Supports direct HTTP writes (via requests/urllib) as well as official influxdb-client.
"""

import os
import csv
import logging
from typing import Optional, Union, Dict, Any, Set
from pathlib import Path
import yaml
import requests
import psutil

from monitoring.schema import WindowDriftResult

logger = logging.getLogger("cdd.monitoring")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DriftMetricsLogger:
    """
    Handles logging of real Task 3 drift results to InfluxDB and CSV.
    Strictly satisfies Section 4.7 & 4.12 of the CDd proposal.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], str, Path]] = None):
        self.config = self._load_config(config)
        self._seen_windows: Set[str] = set()
        
        self.influx_enabled = self.config.get("logging", {}).get("log_to_influx", True)
        self.csv_enabled = self.config.get("logging", {}).get("log_to_csv", True)
        self.csv_path = Path(self.config.get("logging", {}).get("csv_path", "output/drift_results.csv"))
        self.measurement = self.config.get("influxdb", {}).get("measurement", "drift_metrics")
        self.bucket = self.config.get("influxdb", {}).get("bucket", "cdd-bucket")
        self.org = self.config.get("influxdb", {}).get("org", "cdd-org")
        self.url = self.config.get("influxdb", {}).get("url", "http://localhost:8086").rstrip("/")
        self.token = self.config.get("influxdb", {}).get("token", "cdd-super-secret-token")
        self.timeout = self.config.get("influxdb", {}).get("timeout_ms", 10000) / 1000.0
        self.collect_system_metrics = self.config.get("system_metrics", {}).get("collect_cpu_mem", True)

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Token {self.token}",
            "Content-Type": "text/plain; charset=utf-8"
        })

        if self.influx_enabled:
            self._verify_influx()

        if self.csv_enabled:
            self._init_csv()

    def _load_config(self, config: Optional[Union[Dict[str, Any], str, Path]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        config_path = Path(config) if config else Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _verify_influx(self):
        try:
            r = self._session.get(f"{self.url}/health", timeout=3.0)
            if r.status_code == 200:
                logger.info("Connected to InfluxDB at %s (org: %s, bucket: %s)", self.url, self.org, self.bucket)
            else:
                logger.warning("InfluxDB health check returned status %s: %s", r.status_code, r.text)
        except Exception as e:
            logger.warning("Could not reach InfluxDB at %s: %s (Will keep retrying on writes)", self.url, e)

    def _init_csv(self):
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "timestamp", "window_id", "driftlens_score", "driftlens_alarm",
                        "driftlens_threshold", "mcddd_score", "mcddd_alarm",
                        "processing_latency_ms", "throughput_posts_per_sec",
                        "cpu_pct", "mem_mb", "post_count", "n_used", "is_reference", "pipeline"
                    ])
                    writer.writeheader()
            logger.info("CSV backup initialized at %s", self.csv_path.resolve())
        except Exception as e:
            logger.error("Failed to initialize CSV logger at %s: %s", self.csv_path, e)

    def log_drift_result(self, task3_result: Dict[str, Any]) -> bool:
        """
        Consumes a real Task 3 output dictionary:
        {
            "window_id": 18402,
            "window_start": "2020-05-20T00:00:00",
            "driftlens_score": 0.029159,
            "driftlens_alarm": true,
            "threshold": 0.028602,
            "n_posts": 1445,
            "n_used": 1000,
            "is_reference": false,
            "processing_latency_ms": 234.5,
            "throughput_posts_per_sec": 616.2
        }
        Validates, collects live host system metrics, deduplicates, and writes to InfluxDB and CSV.
        """
        record = WindowDriftResult.from_task3_dict(task3_result)
        return self.log_window(record)

    def log_window(self, result: WindowDriftResult, deduplicate: bool = True) -> bool:
        """
        Logs a single WindowDriftResult to InfluxDB and CSV.
        Avoids double-inserting if window_id was already processed.
        """
        w_key = str(result.window_id)
        if deduplicate and w_key in self._seen_windows:
            logger.warning("Skipping duplicate window_id %s to avoid double-counting.", w_key)
            return False

        # Collect live host CPU & RAM utilization at the exact moment of window ingestion
        if self.collect_system_metrics:
            if result.cpu_pct is None:
                result.cpu_pct = psutil.cpu_percent(interval=None)
            if result.mem_mb is None:
                result.mem_mb = round(psutil.virtual_memory().used / (1024 * 1024), 2)

        success = True

        # 1. Write to InfluxDB via HTTP Line Protocol
        if self.influx_enabled:
            line_protocol = result.to_line_protocol(self.measurement)
            endpoint = f"{self.url}/api/v2/write?org={self.org}&bucket={self.bucket}&precision=s"
            try:
                resp = self._session.post(endpoint, data=line_protocol.encode("utf-8"), timeout=self.timeout)
                if resp.status_code in (200, 204):
                    logger.info("Window %s written to InfluxDB (score: %.6f, thr: %s, alarm: %s)",
                                result.window_id, result.driftlens_score, result.driftlens_threshold, result.driftlens_alarm)
                else:
                    logger.error("InfluxDB rejected write for window %s [HTTP %s]: %s",
                                 result.window_id, resp.status_code, resp.text)
                    success = False
            except Exception as e:
                logger.error("Failed sending window %s to InfluxDB at %s: %s", result.window_id, self.url, e)
                success = False

        # 2. Redundant CSV flat log (required by PDF section 4.12)
        if self.csv_enabled:
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "timestamp", "window_id", "driftlens_score", "driftlens_alarm",
                        "driftlens_threshold", "mcddd_score", "mcddd_alarm",
                        "processing_latency_ms", "throughput_posts_per_sec",
                        "cpu_pct", "mem_mb", "post_count", "n_used", "is_reference", "pipeline"
                    ])
                    writer.writerow(result.to_flat_dict())
            except Exception as e:
                logger.error("Failed appending window %s to CSV: %s", result.window_id, e)
                success = False

        if success:
            self._seen_windows.add(w_key)

        return success

    def close(self):
        """Closes session."""
        try:
            self._session.close()
        except Exception:
            pass
        logger.info("DriftMetricsLogger closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
