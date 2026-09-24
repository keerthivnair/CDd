# Task 4 — Real DriftLens InfluxDB + Grafana Monitoring Pipeline

This module connects the real output of **Task 3 (DriftLens)** to **InfluxDB 2.7** and **Grafana**, providing real-time time-series storage, deduplication, host performance metric tracking (CPU/memory), and interactive dashboards matching Sections 4.7 & 4.8 of the project proposal.

---

## 1. Pipeline Architecture

```
Task 1 (Ingestion):
  COVID-19 Twitter dataset  -->  Kafka topic `social-media-stream`
                                         │
                                         ▼
Task 2 (Spark Streaming):
  Kafka topic  -->  PySpark event-time windowing  -->  Cleaned JSONL windows
                                                             │
                                                             ▼
Task 3 (DriftLens):
  Cleaned windows  -->  Sentence Transformer embeddings  -->  DriftLens FDD + Alarm
                                                                    │
                                    Writes to output/drift/driftlens.jsonl
                                                                    │
                                                                    ▼
Task 4 (Monitoring Layer):
  output/drift/driftlens.jsonl  -->  monitoring.ingest (Deduplicates + live CPU/RAM)
                                              ├──> output/drift_results.csv (Backup)
                                              └──> InfluxDB 2.7 (cdd-bucket)
                                                         │
                                                         ▼
                                                    Grafana 11+
                                    "Concept Drift & System Performance Monitoring"
```

---

## 2. InfluxDB Measurement & Schema Reference

- **Measurement**: `drift_metrics`
- **Bucket**: `cdd-bucket`
- **Organization**: `cdd-org`

| Field / Tag | Type | Source | Description |
|---|---|---|---|
| `_time` | Timestamp | Task 3 `window_start` | Event timestamp of window start |
| `window_id` | Tag | Task 3 `window_id` | Unique identifier (e.g. `18402`) |
| `pipeline` | Tag | `"cdd-stream"` | Pipeline stream identifier |
| `driftlens_alarm_state` | Tag | `"drift"` \| `"normal"` | Status tag for quick filtering |
| `driftlens_score` | Field (float) | Task 3 `driftlens_score` | Real Fréchet Drift Distance (FDD) |
| `driftlens_alarm` | Field (int) | Task 3 `driftlens_alarm` | Binary alarm flag (`0` = normal, `1` = alarm) |
| `driftlens_threshold` | Field (float) | Task 3 `threshold` | p99 reference calibrated alarm boundary |
| `processing_latency_ms` | Field (float) | Task 3 timing | Real execution duration (ms) for that window |
| `throughput_posts_per_sec` | Field (float) | Task 3 / Task 2 | Posts processed per second |
| `cpu_pct` | Field (float) | Host `psutil` | CPU utilization percentage during window write |
| `mem_mb` | Field (float) | Host `psutil` | RAM utilization in MB during window write |
| `post_count` | Field (int) | Task 3 `n_posts` | Total tweets in window |
| `n_used` | Field (int) | Task 3 `n_used` | Sample size evaluated ($\le 1000$) |

---

## 3. Grafana Dashboard Panels

Access dashboard at `http://localhost:3000` (Direct Admin access enabled; no login/password required):
1. **DriftLens Score vs Time**: Plots `driftlens_score` vs `driftlens_threshold`.
2. **Drift Alarms**: Visual state timeline (`NORMAL` vs `DRIFT ALARM`).
3. **Processing Latency**: Window execution latency in ms.
4. **Throughput**: Posts processed per second.
5. **CPU Usage**: Real host CPU % utilization.
6. **Memory Usage**: Real host RAM consumption in MB.

---

## 4. End-to-End Execution Guide (Terminal by Terminal)

Run each command from the repository root (`/Users/satheeshkumar/Documents/GitHub/CDd`):

### Terminal 1 — InfluxDB
```bash
docker compose up influxdb
```
*(Runs InfluxDB 2.7 on port 8086 with infinite retention for historical COVID data)*

### Terminal 2 — Grafana
```bash
docker compose up grafana
```
*(Runs Grafana on port 3000 with pre-provisioned datasource & pre-loaded dashboard)*

### Terminal 3 — Kafka / Task 1 (Data Ingestion)
```bash
python ingestion/main.py --config ingestion/config.yaml --no-delay
```
*(Publishes COVID-19 tweets chronologically into Kafka topic `social-media-stream`)*

### Terminal 4 — Spark / Task 2 (Streaming & Windowing)
```bash
python spark/spark_stream.py --config spark/config.yaml
```
*(Consumes Kafka, cleans text, groups into daily windows in `output/windows`)*

### Terminal 5 — DriftLens / Task 3 (Embeddings & Drift Detection)
```bash
python driftlens/run_driftlens.py --config driftlens/config.yaml --follow
```
*(Embeds windows, calibrates reference, calculates FDD score/alarm, and outputs to `output/drift/driftlens.jsonl`)*

### Terminal 6 — Task 4 (InfluxDB Ingestion)
```bash
python -m monitoring.ingest
```
*(Tails `output/drift/driftlens.jsonl`, attaches live system metrics, deduplicates, and ingests points into InfluxDB)*

---

## 5. Verification & Testing

Run unit tests (no external dependencies required):
```bash
python monitoring/tests.py
```
*(Tests schema validation against real Task 3 output, Line Protocol conversion, deduplication, and CSV backup)*
