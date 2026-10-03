# Task 3 — Drift Detection (DriftLens + MCD-DD)

Takes the cleaned daily tweet windows from Task 2, embeds every tweet with a Sentence Transformer, and runs **two** drift detectors on the **exact same** embeddings so we can compare them fairly.

```
Task 2 windows (cleaned tweets, grouped by day)
        │
        ▼
  Embedding (all-MiniLM-L6-v2, 384-d, L2-normalised)
  one .npy file per window, cached on disk
        │
        ├──────────────────────┐
        ▼                      ▼
    DriftLens               MCD-DD
   (statistical)           (neural)
        │                      │
   FDD score + alarm      MCD score + alarm
        │                      │
        └──────┬───────────────┘
               ▼
     output/drift/driftlens.jsonl  (one JSON line per window, both scores)
```

---

## How do the two methods work?

Both methods answer the same question: *"has the language in today's tweets drifted away from a reference period?"* — but they approach it very differently.

### Shared step: embeddings

Every tweet is converted to a 384-dimensional vector by `all-MiniLM-L6-v2` (a Sentence Transformer). These vectors capture the *meaning* of the text — similar sentences land close together, different ones land far apart. Both detectors receive the **identical** vectors; no re-embedding or separate model.

### DriftLens — "how far has the distribution shifted?"

DriftLens treats a window of tweet embeddings as a **cloud of points** and asks: has this cloud moved or changed shape compared to the reference cloud?

**Offline phase (done once on the reference windows):**

1. **Reference split.** The first 14 windows (~50k tweets) are randomly split 50/50:
   - **Baseline half** → used to learn what "normal" looks like.
   - **Threshold half** → held out, only used to figure out the alarm boundary.

2. **PCA reduction.** 384 dimensions is a lot. PCA compresses the baseline embeddings down to 150 dimensions (keeping ~83% of the information). This makes the statistics more stable and the computation faster.

3. **Baseline statistics.** On the PCA-reduced baseline, compute:
   - **μ_b** = mean vector (the "centre" of the cloud, 150-d)
   - **Σ_b** = covariance matrix (the "shape" of the cloud, 150×150)

4. **Threshold calibration.** Draw 1000 random fake "windows" from the threshold half (each from a single reference day). Score each one against the baseline using the formula below. The 99th percentile of these scores becomes the alarm threshold — meaning roughly 1 in 100 normal windows would alarm by chance.

**Online phase (every new window):**

5. **Sample.** Take up to 1000 tweets from the window (fixed sample size so scores are comparable).

6. **Project.** Apply the same PCA learned from the baseline.

7. **Compute Fréchet Distance (FDD):**

```
FDD = ‖μ_b − μ_w‖² + Tr(Σ_b + Σ_w − 2·√(Σ_b · Σ_w))
       ─────────────   ────────────────────────────────
       mean shift         shape difference
```

   - **First term:** squared distance between the centres. If today's tweets talk about different things, the centres move apart.
   - **Second term:** difference in the covariance matrices. If today's tweets are more spread out, or clustered in a different direction, this grows.
   - If FDD = 0, the two distributions are identical.

8. **Alarm.** If `FDD > threshold` → drift detected.

**In short:** DriftLens fits a Gaussian to each window and measures how different it is from the reference Gaussian, using a well-known statistical distance.

---

### MCD-DD — "can a neural network tell windows apart?"

MCD-DD takes a learning-based approach: it trains a small neural network to recognise *concepts* (what a set of samples "looks like"), and detects drift when consecutive windows look too different to the network.

**The encoder (Deep Sets architecture):**

The network takes a *set* of embeddings (not a single one) and outputs a single vector — a "concept summary" of that set. It uses the Deep Sets architecture:

```
   each embedding → MLP → per-item features
                            │
                       mean pool  (makes it order-independent)
                            │
                          MLP → concept vector (64-d)
```

Because of mean-pooling, the output is the same regardless of the order of inputs — a useful property for sets.

**Training (contrastive learning):**

The encoder is trained to make the concept vectors of *similar* sets close together and *different* sets far apart. For each sub-window of a training window, it creates three kinds of pairs:

| Pair type | What is compared | Expected distance |
|---|---|---|
| **Positive** | Two random samples from the same sub-window | Small (same concept) |
| **Weak negative** | Same sub-window, but small noise added to one | Medium |
| **Strong negative** | Different sub-windows + large noise | Large (different concepts) |

The loss function (InfoNCE) pushes the encoder to:
- minimise distance for positive pairs
- maximise distance for negative pairs

A **gradient penalty** (Lipschitz constraint) keeps the distances bounded so they don't explode.

**Drift detection — Maximum Concept Discrepancy (MCD):**

```
MCD = ‖h(prev_window) − h(curr_window)‖²
```

Where `h(·)` is the encoder's concept vector for a window. If this squared distance exceeds the threshold → drift.

The threshold updates dynamically: it's the 95th percentile of the positive-pair distances seen during training. As the encoder improves, the threshold tightens.

**In short:** MCD-DD learns what a "concept" looks like and flags when consecutive windows disagree too much.

---

### Key differences

| | DriftLens | MCD-DD |
|---|---|---|
| **Approach** | Statistical (Gaussian fit + Fréchet distance) | Neural (learned set encoder + concept distance) |
| **Compares against** | Fixed reference baseline (the first 14 windows) | Previous window (sliding) |
| **Drift meaning** | "This window is far from the starting point" | "This window is different from the last one" |
| **Threshold** | Fixed at calibration (p99 of reference FDDs) | Dynamic (updates every window) |
| **Detects** | Gradual + sudden drift from origin | Window-to-window change points |
| **Dependencies** | numpy, scipy, sklearn (lightweight) | PyTorch (heavier) |
| **Interpretability** | High — mean shift vs shape change | Lower — learned representation |

**They complement each other:** DriftLens catches long-term drift from the reference, MCD-DD catches sudden shifts between consecutive days.

---

## Files

| File | Purpose |
|---|---|
| `run_drift.py` | CLI entry point: loads windows, builds references, runs both detectors, writes output |
| `driftlens.py` | DriftLens core (PCA, Gaussian stats, FDD, calibration, save/load) |
| `mcddd.py` | MCD-DD core (Deep Sets encoder, contrastive training, MCD scoring) |
| `embedder.py` | Sentence Transformer wrapper + per-window `.npy` cache |
| `prepare_windows.py` | Offline utility: converts Kaggle CSVs into cleaned 1-day windows (`windows_all.jsonl`) |
| `config.yaml` | All settings (embedding model, DriftLens params, MCD-DD params) |
| `requirements.txt` | Python dependencies |
| `tests/test_driftlens.py` | 12 DriftLens tests (synthetic data, no model download) |
| `tests/test_mcddd.py` | 10 MCD-DD tests (synthetic 384-d data, no model download) |

---

## How to run

**Prerequisites:** Python 3.10+ and virtual environment activated.

### 1. Install dependencies

```bash
pip install -r drift/requirements.txt
```

### 2. Prepare windows (if not already generated)

If `windows_all.jsonl` is not present, generate it directly from the Kaggle dataset in seconds:
```bash
python drift/prepare_windows.py
```
This produces all 154 daily windows (411,879 tweets) matching the exact Spark cleaning and windowing rules.

### 3. Run the drift pipeline

- **Full batch run (re-uses cached embeddings and saved models):**
  ```bash
  python drift/run_drift.py
  ```
  *What it does:* Loads existing `output/drift/reference.npz` and `output/drift/mcddd.pt`, reuses cached `.npy` embeddings, and completes in seconds.

- **Rebuild / refit from scratch:**
  ```bash
  python drift/run_drift.py --refit
  ```
  *What it does:* Forces both detectors to discard saved models, recalculates PCA and baseline Gaussian stats, and retrains MCD-DD from scratch on the 14 reference windows.

- **Quick test (first 20 windows only):**
  ```bash
  python drift/run_drift.py --refit --max-windows 20
  ```

- **DriftLens only (skip MCD-DD):**
  ```bash
  python drift/run_drift.py --refit --no-mcddd
  ```

- **Streaming mode (polls for live windows from Task 2 Spark Structured Streaming):**
  ```bash
  python drift/run_drift.py --follow
  ```
  *What it does:* Waits for the 14 reference windows, fits both models, and stays running in an infinite loop, scoring newly written windows as Spark emits them.

---

## Verification & Testing

Run unit tests to verify both detectors and downstream monitoring integration (zero external services or GPU needed):

```bash
# Test both DriftLens and MCD-DD detectors (22 tests)
pytest drift/tests/ -v

# Test Task 4 monitoring schema, line protocol, and deduplication (4 tests)
python monitoring/tests.py
```
*(All 26 tests run in ~10 seconds on synthetic data without downloading models.)*

---

## Output: All Windows Compared Side-by-Side

Results are written to `output/drift/driftlens.jsonl`. Every single line contains the simultaneous evaluation of **both** detectors for that day:

```json
{
  "window_id": 18402,
  "window_start": "2020-05-20T00:00:00",
  "driftlens_score": 0.029159,
  "driftlens_alarm": true,
  "threshold": 0.028602,
  "mcddd_score": 0.004521,
  "mcddd_alarm": false,
  "mcddd_threshold": 0.012345,
  "n_posts": 1445,
  "n_used": 1000,
  "is_reference": false,
  "processing_latency_ms": 234.5,
  "throughput_posts_per_sec": 616.2
}
```

| Field | Meaning |
|---|---|
| `window_id` / `window_start` | Unique day identifier and starting timestamp |
| `driftlens_score` | Fréchet distance from the fixed 14-day baseline |
| `driftlens_alarm` | `true` if `driftlens_score > threshold` (p99 alarm) |
| `threshold` | DriftLens baseline alarm boundary for this window size |
| `mcddd_score` | Squared Euclidean distance between consecutive concept vectors |
| `mcddd_alarm` | `true` if `mcddd_score > mcddd_threshold` (dynamic alarm) |
| `mcddd_threshold` | MCD-DD dynamic alarm boundary |
| `n_posts` / `n_used` | Total posts in window vs sample size evaluated |
| `is_reference` | `true` for the 14 baseline training windows |
| `processing_latency_ms` | Processing time for embedding and scoring the window |
| `throughput_posts_per_sec` | Throughput achieved during processing |

---

## Next Steps — Task 4 Monitoring Integration

Task 4 consumes `output/drift/driftlens.jsonl` to provide live time-series storage in **InfluxDB** and interactive **Grafana** dashboards:

1. **Start InfluxDB and Grafana** (via Docker):
   ```bash
   docker compose up -d influxdb grafana
   ```
2. **Run the monitoring ingestion script**:
   ```bash
   python -m monitoring.ingest
   ```
   *What it does:* Tails `output/drift/driftlens.jsonl`, attaches host CPU/RAM metrics, and streams both DriftLens and MCD-DD metrics into InfluxDB bucket `cdd-bucket`.
3. **View the live dashboards**:
   Open `http://localhost:3000` in your browser to view:
   - DriftLens Score vs Threshold over time
   - Drift Alarms state timeline
   - Processing Latency and Throughput
   - Host CPU and Memory utilization

---

## CLI options

| Flag | What it does |
|---|---|
| `--config PATH` | Config file (default: `drift/config.yaml`) |
| `--windows-path PATH` | Override the input path |
| `--max-windows N` | Only process the first N windows (quick test) |
| `--refit` | Rebuild reference from scratch (both detectors) |
| `--follow` | Keep polling for new windows (streaming mode) |
| `--no-mcddd` | Disable MCD-DD, run DriftLens only |

---

## Config reference (`config.yaml`)

```yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # 384-d sentence embeddings
  cache_dir: "output/embeddings"                     # .npy cache per window

driftlens:
  reference_windows: 14       # first N windows = "normal" baseline
  threshold_fraction: 0.5     # 50% baseline, 50% threshold calibration
  pca_components: 150         # PCA output dims (from 384)
  window_sample_size: 1000    # tweets per window for scoring
  n_calibration: 1000         # random windows for threshold estimation
  percentile: 99              # alarm = top 1% of reference distances
  seed: 42

mcddd:
  enabled: true
  hidden_dim: 128             # encoder hidden layer width
  output_dim: 64              # concept vector size
  n_sub_windows: 5            # splits per window for contrastive pairs
  sample_set_size: 100        # items per set in training pairs
  train_epochs: 3             # gradient steps on reference data
  model_path: "output/drift/mcddd.pt"
```

---

## Results on the full dataset

153 daily windows, 410k tweets, 2020-04-19 to 2021-06-26.

**DriftLens:** 122 of 139 post-reference windows alarmed. Drift is gradual — May 2020 stays close to the baseline (median FDD 0.027), then steadily climbs. By 2021, median scores (0.058–0.070) are ~2× higher than mid-2020 (0.036–0.041). Sharpest spike: 2020-10-02 (FDD 0.163) — the day of the US president's COVID-19 diagnosis.

| Month | Windows | Alarms | Median FDD |
|---|---|---|---|
| 2020-05 (post-ref) | 26 | 10 | 0.0267 |
| 2020-06 | 17 | 16 | 0.0363 |
| 2020-08 | 10 | 10 | 0.0414 |
| 2020-09 | 19 | 19 | 0.0358 |
| 2020-10 | 20 | 20 | 0.0398 |
| 2021-04 | 5 | 5 | 0.0695 |
| 2021-05 | 24 | 24 | 0.0610 |
| 2021-06 | 18 | 18 | 0.0575 |

---

## Design decisions

- **Reference = first 14 windows.** Taken as "normal." 14 windows ≈ 50k tweets → stable 150-d covariance.
- **Fixed sample size (1000).** FDD is biased by sample size, so all windows are sampled to 1000 for fair comparison. Smaller windows get size-specific thresholds.
- **p99 threshold.** ≈1 false alarm per 100 normal windows. For daily windows, that's roughly 1 every 3 months.
- **No reference re-fitting.** We want to measure cumulative drift from a fixed starting point, not adapt to it.
- **Both detectors on identical embeddings.** Fair comparison — any difference in alarms is due to the detection method, not the input representation.
