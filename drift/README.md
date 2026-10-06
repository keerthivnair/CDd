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

MCD-DD ([Wan et al., KDD 2024](https://arxiv.org/abs/2407.05375)) takes a learning-based approach: it trains a small neural network to recognise *concepts* (what a set of samples "looks like"), and detects drift when the newest window looks too different from the recent windows before it.

**The encoder (Deep Sets architecture):**

The network takes a *set* of embeddings (not a single one) and outputs a single vector — a "concept summary" of that set. It uses the Deep Sets architecture:

```
   each embedding → MLP → per-item features
                            │
                       mean pool  (makes it order-independent)
                            │
                          MLP → concept vector (64-d)
```

`n_encoders` (default 5) such encoders are initialised and trained independently, and the concept vector is their concatenation (÷√5). A single encoder's results depended heavily on its random start, and the ensemble averages that out.

Because of mean-pooling, the output is the same regardless of the order of inputs — a useful property for sets.

**Training (contrastive learning, offline on the reference period):**

Inputs are first standardised with the reference mean/std. Each reference window (one day) is a *sub-window*; 20% of its tweets train the encoder and 80% are **held out**. The encoder is trained to make the concept vectors of *similar* sets close together and *different* sets far apart. For each sub-window it draws `k_pairs` fresh pairs per step (100 steps):

| Pair type | What is compared | Expected distance |
|---|---|---|
| **Positive** | Two random 100-tweet samples from the same day | Small (same concept) |
| **Weak negative** | Same day, small Gaussian noise (`eps_small`) added to one | Medium |
| **Strong negative** | A temporally distant day + larger noise (`eps_big`) | Large (different concepts) |

The InfoNCE loss pulls positives together and pushes negatives apart. A **gradient penalty** (paper's λ = 1) keeps the encoder L-Lipschitz so distances stay bounded.

**Drift detection — Maximum Concept Discrepancy (MCD):**

```
MCD_t = max over the previous C windows j of ‖h(window_t) − h(window_j)‖₂      (C = context_windows = 10)
```

`h(·)` is the encoder's concept vector of a random ≤1000-tweet sample of the window (same size as DriftLens, so traffic volume does not change the noise level). Comparing against the whole sliding context — as in the paper's Fig. 3 heatmaps — is what lets *gradual* drift build up into a detectable gap; comparing only with the previous day cannot see a change of a few percent per day.

**Threshold (frozen after fit):** bootstrap on the held-out reference tweets — draw C+1 in-distribution sets of the scoring size, compute the same max-MCD statistic, repeat 500×, take the 99th percentile. It is recomputed (and cached) for smaller windows or a shorter context, the way DriftLens recalibrates per window size.

**In short:** MCD-DD learns what a "concept" looks like and flags when today disagrees too much with any of the last 10 days.

**Why these choices (validated on separate streams, not the official scenarios):**

| Problem found | Effect | Fix |
|---|---|---|
| Pair RNG was reseeded on every step | All "epochs" trained on one identical batch | One persistent generator; fresh pairs each step |
| Threshold calibrated on 100-tweet sets, score on whole windows; threshold re-set every window | Threshold ≈ 40× too high by the end of a stream, 0 alarms | Calibrate at the scoring set size with the scoring statistic; freeze it |
| Calibration/context used the encoder's own training tweets | Encoder memorises them → every window next to the reference alarmed | 80/20 hold-out per reference window (enough held-out tweets for 11 disjoint 1000-tweet calibration sets) |
| `eps_big` = 1.0 (paper-style large noise) | Encoder learns "spot the noise", sensitivity to topic change drops sharply | `eps_small` 0.03, `eps_big` 0.3 (paper's 1:10 ratio) |
| Raw MiniLM dims have std ≈ 0.05 | MCD values ~1e-6, rounded to 0 in the output | Standardise inputs; L2 (not squared) distance, as in the paper |
| One encoder per run | Results swung with the random start (gradual detected in 2/5 seeds) | Ensemble of 5 independently trained encoders |
| torch 2.6+cu124 lacks kernels for RTX 50xx (sm_120) | Silent CPU fallback | Install a cu128 build (see below); device picked by actually running a kernel |

**Trained vs. untrained encoder.** On this data the reference days are one stationary concept, so contrastive training has no real concept change to learn from. An untrained ensemble (`train_epochs: 0`) detected drift better in our experiments (see `experiments/README.md`). The trained variant is the default because it follows the paper.

Deviations from the paper: the encoder is trained offline on the reference only (`online_training: false`; online updates are supported but inflated the threshold in earlier runs), and the threshold is bootstrapped on held-out reference data rather than from the latest positive pairs.

---

### Key differences

| | DriftLens | MCD-DD |
|---|---|---|
| **Approach** | Statistical (Gaussian fit + Fréchet distance) | Neural (learned set encoder + concept distance) |
| **Compares against** | Fixed reference baseline (the first 14 windows) | Sliding context of the previous 10 windows |
| **Drift meaning** | "This window is far from the starting point" | "This window differs from some recent window" |
| **Threshold** | Fixed at calibration (p99 of reference FDDs) | Fixed at calibration (p99 of bootstrapped max-MCD) |
| **Detects** | Gradual + sudden drift from origin | Change points; alarms stop ~10 windows after a drift once the context has adapted |
| **Dependencies** | numpy, scipy, sklearn (lightweight) | PyTorch (heavier) |
| **Interpretability** | High — mean shift vs shape change | Lower — learned representation |

**They complement each other:** DriftLens says "we are far from where we started" (and keeps alarming), MCD-DD says "something changed in the last ~10 days" (and goes quiet once the new concept is the norm).

---

## Files

| File | Purpose |
|---|---|
| `run_drift.py` | CLI entry point: loads windows, builds references, runs both detectors, writes output |
| `driftlens.py` | DriftLens core (PCA, Gaussian stats, FDD, calibration, save/load) |
| `mcddd.py` | MCD-DD core (Deep Sets encoder, contrastive training, MCD scoring) |
| `embedder.py` | Sentence Transformer wrapper + per-window `.npy` cache |
| `compare_detectors.py` | Side-by-side DriftLens vs MCD-DD per window → `comparison.jsonl` + `comparison_summary.json` |
| `prepare_windows.py` | Offline utility: converts Kaggle CSVs into cleaned 1-day windows (`windows_all.jsonl`) |
| `config.yaml` | All settings (embedding model, DriftLens params, MCD-DD params) |
| `requirements.txt` | Python dependencies |
| `tests/test_driftlens.py` | 12 DriftLens tests (synthetic data, no model download) |
| `tests/test_mcddd.py` | 14 MCD-DD tests (synthetic 384-d data, no model download) |

---

## How to run

**Prerequisites:** Python 3.10+ and virtual environment activated.

### 1. Install dependencies

```bash
pip install -r drift/requirements.txt
```

**GPU (RTX 50-series / Blackwell, sm_120):** the default `torch` wheel built for CUDA 12.4 has no sm_120 kernels, so everything silently runs on the CPU. Install a CUDA 12.8 build instead (needs NVIDIA driver ≥ 570):
```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```
Check: `python -c "import torch; print(torch.cuda.get_arch_list())"` must list `sm_120`. The run log line `MCD-DD fit on cuda` / `Loaded Sentence Transformer ... on cuda:0` confirms the GPU is used.

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
# Test both DriftLens and MCD-DD detectors (26 tests)
pytest drift/tests/ -v

# Test Task 4 monitoring schema, line protocol, and deduplication (4 tests)
python monitoring/tests.py
```
*(All 30 tests run in ~10 seconds on synthetic data without downloading models.)*

---

## Output: All Windows Compared Side-by-Side

For a cleaner comparison file, run `python drift/compare_detectors.py` (or `--scores output/experiments/<name>/scores.jsonl`). It writes `comparison.jsonl` next to the scores. Each row holds both detectors' score, threshold and **score/threshold ratio** (the raw scores have different units, while the ratios are comparable, >1 = alarm), plus `agreement`: `both` / `driftlens_only` / `mcddd_only` / `neither`. A monthly alarm summary goes to `comparison_summary.json`.

Results are written to `output/drift/driftlens.jsonl`. Every single line contains the simultaneous evaluation of **both** detectors for that day:

```json
{
  "window_id": 18402,
  "window_start": "2020-05-20T00:00:00",
  "driftlens_score": 0.029159,
  "driftlens_alarm": true,
  "threshold": 0.028602,
  "mcddd_score": 1.84,
  "mcddd_alarm": false,
  "mcddd_threshold": 2.31,
  "mcddd_lag": 6,
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
| `mcddd_score` | Max L2 distance between this window's concept vector and those of the previous 10 windows (`null` for reference windows, which are MCD-DD's training data) |
| `mcddd_alarm` | `true` if `mcddd_score > mcddd_threshold` (p99 alarm) |
| `mcddd_threshold` | MCD-DD alarm boundary for this window size and context length |
| `mcddd_lag` | How many windows back the largest discrepancy was (1 = previous day) |
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
  device: ""                  # "" = cuda if usable, else cpu
  hidden_dim: 128             # encoder hidden layer width
  output_dim: 64              # concept vector size
  n_encoders: 5               # independently trained encoders (ensemble)
  lambda_gp: 1.0              # gradient penalty (paper default)
  eps_small: 0.03             # weak-negative noise (standardised units)
  eps_big: 0.3                # strong-negative noise (1:10 ratio)
  sample_set_size: 100        # items per set in training pairs
  k_pairs: 10                 # pairs per day per step
  train_epochs: 100           # gradient steps on reference data (seconds on GPU)
  scoring_sample_size: 1000   # tweets per window for scoring
  context_windows: 10         # newest window vs. the previous 10 (max MCD)
  holdout_fraction: 0.8       # reference tweets kept out of training, used for calibration
  n_calibration: 500          # bootstrap draws for the threshold
  percentile: 99              # alarm = top 1% of in-distribution max-MCD
  model_path: "output/drift/mcddd.pt"
```

---

## Results on the full dataset

154 daily windows, 412k tweets, 2020-04-19 to 2021-06-27 (rerun 2026-10-06 with the fixed MCD-DD; per-window results in `output/drift/driftlens.jsonl`, side by side in `output/drift/comparison.jsonl`).

**DriftLens:** 120 of 140 post-reference windows alarmed. Drift is gradual: May 2020 stays close to the baseline (median FDD 0.027), then steadily climbs. By 2021, median scores (0.058–0.076) are ~2× higher than mid-2020 (0.036–0.041). Sharpest spike: 2020-10-02 (FDD 0.164), the day of the US president's COVID-19 diagnosis.

**MCD-DD:** 12 of 140 windows alarmed. It flags *recent* changes rather than distance from the start: early May 2020, mid-October 2020 (the days after the 2020-10-02 spike), 2021-04-26 (the first day of 2021 data, after a gap of several months) and late June 2021. 10 of its 12 alarms are also DriftLens alarms.

| Month | Windows | DriftLens alarms | Median FDD | MCD-DD alarms |
|---|---|---|---|---|
| 2020-05 (post-ref) | 26 | 7 | 0.0271 | 2 |
| 2020-06 | 17 | 16 | 0.0363 | 0 |
| 2020-08 | 10 | 10 | 0.0409 | 0 |
| 2020-09 | 19 | 19 | 0.0365 | 0 |
| 2020-10 | 20 | 20 | 0.0395 | 5 |
| 2021-04 | 5 | 5 | 0.0756 | 1 |
| 2021-05 | 24 | 24 | 0.0603 | 0 |
| 2021-06 | 19 | 19 | 0.0575 | 4 |

---

## Design decisions

- **Reference = first 14 windows.** Taken as "normal." 14 windows ≈ 50k tweets → stable 150-d covariance.
- **Fixed sample size (1000).** FDD is biased by sample size, so all windows are sampled to 1000 for fair comparison. Smaller windows get size-specific thresholds.
- **p99 threshold.** ≈1 false alarm per 100 normal windows. For daily windows, that's roughly 1 every 3 months.
- **No reference re-fitting.** We want to measure cumulative drift from a fixed starting point, not adapt to it.
- **Both detectors on identical embeddings.** Fair comparison — any difference in alarms is due to the detection method, not the input representation.
