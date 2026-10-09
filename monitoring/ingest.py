"""
Task 4 Ingestion Runner: InfluxDB + Grafana Monitoring Layer.

Default Mode:
  Consumes real output produced by DriftLens & MCD-DD (from output/drift/driftlens.jsonl
  or user-specified file) and writes all metrics, agreement categories, and performance
  statistics to InfluxDB and CSV.

Features:
  - Supports --source to ingest from any JSONL file (e.g. Downloads/driftlens.jsonl)
  - Supports --reset-state to force re-ingestion of historical runs
  - Automatically classifies detector agreement (both, driftlens_only, mcddd_only, neither)
  - Avoids duplicate records with state tracking
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Set, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.client import DriftMetricsLogger
from monitoring.schema import WindowDriftResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("cdd.ingest")


def ingest_drift_file(
    logger: DriftMetricsLogger,
    file_path: Path,
    follow: bool = False,
    poll_interval: float = 1.0,
    reset_state: bool = False,
    state_file: Path = Path("output/drift/ingested_state.txt")
) -> Dict[str, int]:
    """
    Ingests detector results from driftlens.jsonl into InfluxDB and CSV.
    Tails the file if follow=True, and deduplicates by window_id.
    """
    log.info("Processing detector results from: %s", file_path.resolve())

    if reset_state and state_file.exists():
        try:
            state_file.unlink()
            log.info("Reset state file: %s", state_file)
        except Exception as e:
            log.warning("Could not delete state file: %s", e)

    ingested_windows: Set[str] = set()
    if state_file.exists():
        try:
            with open(state_file, "r") as sf:
                for line in sf:
                    w = line.strip()
                    if w:
                        ingested_windows.add(w)
            log.info("Loaded %s previously ingested window IDs from state file.", len(ingested_windows))
        except Exception as e:
            log.warning("Could not load state file: %s", e)

    while not file_path.exists():
        if not follow:
            log.error("File %s does not exist. Run Task 3 first or check the path.", file_path)
            return {}
        log.info("Waiting for detector output file %s...", file_path)
        time.sleep(poll_interval)

    state_file.parent.mkdir(parents=True, exist_ok=True)
    summary_counts = {
        "total": 0,
        "new_ingested": 0,
        "skipped_duplicate": 0,
        "both": 0,
        "driftlens_only": 0,
        "mcddd_only": 0,
        "neither": 0,
        "reference": 0,
        "errors": 0,
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f, open(state_file, "a", encoding="utf-8") as sf:
            while True:
                line = f.readline()
                if line:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    summary_counts["total"] += 1
                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError as err:
                        log.error("Corrupt JSON line: %s (Error: %s)", line_str[:60], err)
                        summary_counts["errors"] += 1
                        continue

                    w_id = str(record.get("window_id"))
                    if w_id in ingested_windows:
                        summary_counts["skipped_duplicate"] += 1
                        continue

                    try:
                        parsed_rec = WindowDriftResult.from_task3_dict(record)
                        success = logger.log_window(parsed_rec, deduplicate=True)
                        if success:
                            ingested_windows.add(w_id)
                            sf.write(f"{w_id}\n")
                            sf.flush()
                            summary_counts["new_ingested"] += 1
                            agree = parsed_rec.agreement
                            if agree in summary_counts:
                                summary_counts[agree] += 1
                        else:
                            summary_counts["errors"] += 1
                    except ValueError as ve:
                        log.error("Validation error for window %s: %s", w_id, ve)
                        summary_counts["errors"] += 1
                else:
                    if not follow:
                        log.info("Finished reading %s.", file_path)
                        break
                    time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Ingestion interrupted by user.")

    log.info(
        "Ingestion summary: %d total, %d newly ingested, %d duplicates skipped, %d errors. "
        "Agreements: both=%d, DL_only=%d, MCD_only=%d, neither=%d, reference=%d",
        summary_counts["total"], summary_counts["new_ingested"], summary_counts["skipped_duplicate"],
        summary_counts["errors"], summary_counts["both"], summary_counts["driftlens_only"],
        summary_counts["mcddd_only"], summary_counts["neither"], summary_counts["reference"]
    )
    return summary_counts


def main():
    parser = argparse.ArgumentParser(description="Task 4: InfluxDB Drift Results Ingestor")
    parser.add_argument("--config", default="monitoring/config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", default="output/drift/driftlens.jsonl", help="Path to driftlens.jsonl detector output")
    parser.add_argument("--follow", action="store_true", help="Keep tailing file for live streaming results")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--reset-state", action="store_true", help="Clear state file and re-ingest all records")
    args = parser.parse_args()

    logger = DriftMetricsLogger(config=args.config)

    try:
        source_file = Path(args.source)
        if not source_file.is_absolute():
            source_file = PROJECT_ROOT / source_file
        ingest_drift_file(
            logger=logger,
            file_path=source_file,
            follow=args.follow,
            poll_interval=args.poll_interval,
            reset_state=args.reset_state,
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
