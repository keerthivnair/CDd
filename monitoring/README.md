# Task 4 / Person 4: Drift Explainability + Grafana Monitoring Pipeline

This module implements the **Person 4: Drift Explainability + Grafana Dashboard Extension** for the CDd COVID-19 streaming project. It connects real detector results from **DriftLens (Task 3)** and **MCD-DD** to **InfluxDB 2.7** and **Grafana**, provides a **10-panel monitoring dashboard**, and features a full **Drift Explainability Engine** that uncovers why detectors alarmed, highlights semantic shifts, and analyzes detector disagreements.

---

## 1. End-to-End Pipeline Architecture

```
Task 1 (Ingestion):
  COVID-19 Twitter dataset  -->  Kafka topic `social-media-stream`
                                         │
                                         ▼
Task 2 (Spark Streaming):
  Kafka topic  -->  PySpark event-time windowing  -->  Cleaned JSONL windows
                                                              │
                                                              ▼
Task 3 (DriftLens & MCD-DD):
  Cleaned windows  -->  Sentence Transformer embeddings  -->  DriftLens (FDD) & MCD-DD Detectors
                                                                    │
                                    Writes to output/drift/driftlens.jsonl
                                                                    │
                                                                    ▼
Task 4 / Person 4 (Monitoring & Explainability):
  output/drift/driftlens.jsonl  ──> monitoring.ingest (Deduplicates + live CPU/RAM)
                                            ├──> output/drift_results.csv (Backup CSV)
                                            └──> InfluxDB 2.7 (cdd-bucket)
                                                       │
                                                       ▼
                                            Grafana 11+ (10 Panels)
                                            "Concept Drift & System Performance Monitoring"

                                            AND

  output/drift/driftlens.jsonl  ──> monitoring.explain (Drift Explainer Engine)
                                            ├──> output/drift/disagreement_table.csv
                                            ├──> output/drift/explainability_report.json
                                            └──> monitoring/explainability_viewer.html (Interactive Web UI)
```

---

## 2. InfluxDB Schema & Measurement Reference

- **Measurement**: `drift_metrics`
- **Bucket**: `cdd-bucket`
- **Organization**: `cdd-org`
- **Retention**: Infinite (`0`) to support historical COVID-19 event timestamps

### Tags & Fields

| Key | Type | Category | Description |
|---|---|---|---|
| `_time` | Timestamp | Timestamp | Window start event timestamp |
| `window_id` | Tag | Metadata | Unique window identifier (e.g. `18402`) |
| `pipeline` | Tag | Metadata | Stream identifier (`cdd-stream`) |
| `agreement` | Tag | Classification | Detector agreement (`both`, `driftlens_only`, `mcddd_only`, `neither`, `reference`) |
| `driftlens_alarm_state` | Tag | Status | Binary alarm tag (`drift` vs `normal`) |
| `mcddd_alarm_state` | Tag | Status | Binary MCD-DD alarm tag (`drift` vs `normal`) |
| `is_reference` | Tag | Baseline | Whether window belongs to reference calibration (`true` vs `false`) |
| `driftlens_score` | Field (float) | Metric | Fréchet Drift Distance (FDD) score |
| `driftlens_threshold` | Field (float) | Threshold | Calibrated p99 alarm threshold for DriftLens |
| `driftlens_alarm` | Field (int) | Alarm | Binary alarm (`1` = alarm, `0` = normal) |
| `driftlens_ratio` | Field (float) | Normalized | Ratio of score to threshold (`driftlens_score / threshold`) |
| `mcddd_score` | Field (float) | Metric | Minimum Covariance Determinant drift score |
| `mcddd_threshold` | Field (float) | Threshold | Calibrated threshold for MCD-DD |
| `mcddd_alarm` | Field (int) | Alarm | Binary alarm (`1` = alarm, `0` = normal) |
| `mcddd_ratio` | Field (float) | Normalized | Ratio of score to threshold (`mcddd_score / mcddd_threshold`) |
| `agreement_code` | Field (int) | Mapping | Integer code for Grafana timeline (`0`: ref, `1`: neither, `2`: dl_only, `3`: mcd_only, `4`: both) |
| `processing_latency_ms` | Field (float) | Performance | Real processing duration in milliseconds |
| `throughput_posts_per_sec` | Field (float) | Performance | Posts processed per second |
| `n_posts` / `post_count` | Field (int) | Volume | Total raw tweets in window |
| `n_used` | Field (int) | Volume | Number of tweets sampled/evaluated ($\le 1000$) |
| `cpu_pct` | Field (float) | Host | Host CPU utilization percentage |
| `mem_mb` | Field (float) | Host | Host RAM consumption in MB |

---

## 3. Grafana 10-Panel Dashboard

The dashboard is auto-provisioned at `http://localhost:3000` (Direct Admin access enabled, no password required) and organized into three structured sections:

### Section 1: Concept Drift Detectors & Alarm Analysis
- **Panel 1: DriftLens Score vs Time**: Timeseries plotting `driftlens_score` against dashed `driftlens_threshold`.
- **Panel 2: MCD-DD Score vs Time**: Independent timeseries plotting `mcddd_score` against dashed `mcddd_threshold` on its native metric scale.
- **Panel 3: Detector Agreement & Alarm States**: State timeline visualizing `agreement_code`, `driftlens_alarm`, and `mcddd_alarm` with clear color coding.
- **Panel 4: Drift Alarm Investigation & Disagreement Table**: Interactive table pivoting window records with scores, normalized ratios, and color-coded agreement tags for operator triage.

### Section 2: Tweet Stream Characteristics & Explainability
- **Panel 5: Tweet Volume (Total vs Used Posts)**: Timeseries comparing total ingested window posts (`n_posts`) vs sampled posts (`n_used`).
- **Panel 10: Drift Explainability & Semantic Shift Portal**: Rich markdown documentation summarizing consensus findings, key regime shifts, and portal launch instructions.

### Section 3: System Performance & Resource Metrics
- **Panel 6: Processing Latency**: Real window computation duration in ms.
- **Panel 7: Processing Throughput**: Execution throughput in posts/sec.
- **Panel 8: CPU Utilization**: Host CPU load % with alert thresholds.
- **Panel 9: Memory Utilization**: Host memory consumption in MB.

---

## 4. Drift Explainability Engine & Standalone Web Viewer

Whenever concept drift occurs, operators must understand **what actually changed**. The explainability engine (`monitoring/explain.py`) performs:

1. **Disagreement Prioritization**: Isolates cases where detectors diverge (e.g. `mcddd_only` in early May 2020 vs `driftlens_only` gradual trends).
2. **Semantic Theme & Keyword Shift Analysis**: Evaluates TF-IDF n-gram frequency growth and drops to identify *emerging* (newly surging) and *fading* (cooling) terms.
3. **Representative Tweet Extraction**: Computes cosine distance to window embedding centroids to pick 3 representative tweets instead of overwhelming operators with raw tweet streams.
4. **Missing Data Honesty**: If raw tweets (`windows_all.jsonl`) are not present locally, the system logs an explicit notice without fabricating fictional tweets, and uses verified historical traces in the companion UI.

### Launching the Standalone Explainability Viewer

```bash
# From repository root:
python3 -m http.server 5050
```

Open in your browser:
**`http://localhost:5050/monitoring/explainability_viewer.html`**

The viewer provides:
- KPI summary cards (Total Windows, Both Alarms, Disagreements)
- Before vs After comparison cards for key drift windows
- Emerging & Fading keyword tag clouds
- Searchable, filterable disagreement table with direct inspection links

---

## 5. End-to-End Terminal Execution Guide

Run each command from the repository root (`/Users/satheeshkumar/Documents/GitHub/CDd`):

### Terminal 1 — InfluxDB Service
```bash
docker compose up influxdb
```
*(Runs InfluxDB 2.7 on port 8086 with infinite retention)*

### Terminal 2 — Grafana Dashboard
```bash
docker compose up grafana
```
*(Runs Grafana on port 3000 with pre-provisioned datasource & 10-panel dashboard)*

### Terminal 3 — Kafka Data Ingestion (Task 1)
```bash
python ingestion/main.py --config ingestion/config.yaml --no-delay
```

### Terminal 4 — Spark Windowing (Task 2)
```bash
python spark/spark_stream.py --config spark/config.yaml
```

### Terminal 5 — Drift Detection (Task 3)
```bash
python drift/run_drift.py --config drift/config.yaml --follow
```

### Terminal 6 — Task 4 InfluxDB Ingestion
```bash
python -m monitoring.ingest --source output/drift/driftlens.jsonl
```
*(Reads detector results, computes agreement tags and ratios, and writes to InfluxDB + CSV)*

### Terminal 7 — Explainability Engine & Web Viewer
```bash
python monitoring/explain.py
python3 -m http.server 5050
```
*(Generates `explainability_report.json` and `disagreement_table.csv`, then serves `explainability_viewer.html`)*

---

## 6. Unit & Integration Testing

Run the test suite using Python's standard `unittest`:
```bash
python monitoring/tests.py
```

Covers:
- Real Task 3 dictionary parsing
- All 5 agreement classification branches
- MCD-DD score and threshold ratio validation
- Line protocol serialization with agreement tags
- InfluxDB deduplication protection
- TF-IDF emerging/fading keyword extraction
- Important window selection
- Honest missing raw tweet handling
