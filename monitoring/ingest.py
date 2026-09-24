"""
Task 4 Ingestion Runner: InfluxDB + Grafana Monitoring Layer.

Default Mode:
  Continuously tails and consumes the real output produced by Task 3 (DriftLens)
  from output/drift/driftlens.jsonl and writes it directly to InfluxDB and CSV.

Optional Dev Mode:
  --simulate: Generates sample windows for offline testing without running Kafka/Spark.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Set

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.client import DriftMetricsLogger
from monitoring.schema import WindowDriftResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("cdd.ingest")


def follow_driftlens_output(
    logger: DriftMetricsLogger,
    file_path: Path,
    follow: bool = True,
    poll_interval: float = 1.0,
    state_file: Path = Path("output/drift/ingested_state.txt")
):
    """
    Consumes real DriftLens results from output/drift/driftlens.jsonl.
    Tails the file as Task 3 writes and deduplicates by window_id.
    """
    log.info("Monitoring real Task 3 output at: %s", file_path.resolve())

    # Load previously ingested window IDs if state file exists
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
            log.error("File %s does not exist. Run Task 3 (driftlens/run_driftlens.py) first.", file_path)
            return
        log.info("Waiting for Task 3 to create %s...", file_path)
        time.sleep(poll_interval)

    file_pos = 0
    state_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "r", encoding="utf-8") as f, open(state_file, "a", encoding="utf-8") as sf:
            while True:
                line = f.readline()
                if line:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError as err:
                        log.error("Corrupt JSON line in %s: %s (Error: %s)", file_path, line_str[:60], err)
                        continue

                    w_id = str(record.get("window_id"))
                    if w_id in ingested_windows:
                        continue

                    # Validate and log to InfluxDB + CSV
                    try:
                        success = logger.log_drift_result(record)
                        if success:
                            ingested_windows.add(w_id)
                            sf.write(f"{w_id}\n")
                            sf.flush()
                            log.info("Ingested window %s (score: %s, alarm: %s)",
                                     w_id, record.get("driftlens_score"), record.get("driftlens_alarm"))
                    except ValueError as ve:
                        log.error("Validation error for window %s: %s", w_id, ve)

                else:
                    if not follow:
                        log.info("Finished processing all lines in %s.", file_path)
                        break
                    time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Stopping ingestion process upon user interrupt.")


def run_dev_simulation(logger: DriftMetricsLogger, count: int = 30, interval: float = 0.5):
    """Developer testing only: simulates streaming windows for offline dashboard checks."""
    import random
    log.warning("Running DEV SIMULATION mode. (This is for offline testing only, NOT for real pipeline).")
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(minutes=count * 2)

    for i in range(1, count + 1):
        window_time = start_time + timedelta(minutes=i * 2)
        window_id = 18350 + i
        score = random.uniform(0.015, 0.045) if i <= 15 else random.uniform(0.045, 0.085)
        thr = 0.0286
        alarm = bool(score > thr)
        latency = random.uniform(120.0, 320.0)
        posts = random.randint(400, 1200)
        throughput = round(posts / (latency / 1000.0), 1)

        sim_record = {
            "window_id": window_id,
            "window_start": window_time.isoformat(),
            "driftlens_score": round(score, 6),
            "driftlens_alarm": alarm,
            "threshold": thr,
            "n_posts": posts,
            "n_used": min(posts, 1000),
            "is_reference": (i <= 14),
            "processing_latency_ms": round(latency, 2),
            "throughput_posts_per_sec": throughput,
        }
        logger.log_drift_result(sim_record)
        if interval > 0 and i < count:
            time.sleep(interval)
    log.info("Simulation completed.")


def main():
    parser = argparse.ArgumentParser(description="Task 4: InfluxDB + Grafana Real DriftLens Results Ingestor")
    parser.add_argument("--config", default="monitoring/config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", default="output/drift/driftlens.jsonl", help="Path to Task 3 driftlens.jsonl output")
    parser.add_argument("--no-follow", action="store_true", help="Process existing records and exit (do not tail)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--simulate", action="store_true", help="[DEV ONLY] Run simulation instead of reading Task 3")
    args = parser.parse_args()

    logger = DriftMetricsLogger(config=args.config)

    try:
        if args.simulate:
            run_dev_simulation(logger)
        else:
            source_file = Path(args.source)
            if not source_file.is_absolute():
                source_file = PROJECT_ROOT / source_file
            follow_driftlens_output(
                logger=logger,
                file_path=source_file,
                follow=not args.no_follow,
                poll_interval=args.poll_interval
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
