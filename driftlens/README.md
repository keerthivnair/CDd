# Task 3 - Embeddings + DriftLens drift detection

**What this folder does:** takes the cleaned daily tweet windows produced by Task 2, embeds every tweet with a Sentence Transformer, and uses DriftLens to give each window a drift score and an alarm when its embedding distribution has moved away from a reference period.

```
Task 2 windows (windows_all.jsonl, or output/windows/batch_id=*/part-*.json)
        |
        |  1. load windows, sorted by window_id
        |  2. embed tweets: all-MiniLM-L6-v2 (384-d, L2-normalised), cached per window
        |  3. reference = first 14 windows; split 50/50 into baseline / threshold pool
        |  4. PCA (150 dims) fitted on the baseline
        |  5. baseline mean + covariance in PCA space
        |  6. per window: sample <= 1000 tweets, project, mean + covariance
        |  7. Frechet distance (FDD) window vs. baseline  -> driftlens_score
        |  8. threshold = p99 of FDDs of in-distribution windows of the same size
        |  9. alarm = score > threshold
        v
output/drift/driftlens.jsonl (one line per window) + output/drift/reference.npz
```

## What was implemented

| Step | What it does |
|---|---|
| Input | One JSONL file or Task 2's folder output. Windows are keyed and sorted by `window_id`, so a retried Spark batch that rewrote a window does not produce duplicates. |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, `normalize_embeddings=True`. The only extra text step is removing U+FFFD characters left by encoding damage in the source CSVs; Task 2 already did the real cleaning. |
| Device | `device: ""` = auto: CUDA when PyTorch sees a GPU, otherwise CPU. `"cpu"` / `"cuda"` force one. |
| Embedding cache | One `.npy` per window in `output/embeddings/<model>/`. Re-used when its row count matches the window, so re-runs and threshold changes do not re-embed. |
| Reference | The first `reference_windows` (14) windows present in the stream. These are windows, not calendar days: the data has gaps, so the 14 windows cover 2020-04-19 to 2020-05-04. |
| Baseline / threshold split | Reference tweets are randomly split 50/50. PCA and the baseline mean/covariance use one half; the other half is only used for calibration, so the threshold is not measured on the data it was fitted to. |
| PCA | 150 components fitted on the baseline half (keeps 83.5% of the variance on this data). Stored as plain arrays so a saved reference reloads with identical results. |
| Drift score | Frechet distance between Gaussians: `||mu_b - mu_w||^2 + Tr(S_b + S_w - 2 sqrt(S_b S_w))`. |
| Window sample size | Each window is scored on a deterministic random sample of at most 1000 tweets (seeded by `window_id`). FDD grows as the sample shrinks, so windows must be compared at a fixed size. |
| Threshold | The 99th percentile of FDDs of 1000 in-distribution windows drawn from the threshold pool at the same size. Each calibration window is drawn from a **single** reference day (grouped calibration), so the threshold covers normal day-to-day variation, not just the smaller distance of a random mix of all 14 days. |
| Small windows | A window with fewer than 1000 tweets gets its own threshold calibrated at its size (cached per size). If no reference day is large enough, it falls back to the pooled threshold set. |
| Persistence | `reference.npz` stores PCA, baseline stats, threshold pool and all calibrated thresholds. It is reused on the next run only if the model, reference window ids and all parameters match; `--refit` forces a rebuild. |
| Streaming mode | `--follow` polls the input every `poll_seconds` and scores new windows as Task 2 writes them. It waits until the reference period is complete. |

## Files

| File | Purpose |
|---|---|
| `run_driftlens.py` | CLI: loading windows, reference build/reuse, scoring loop, output |
| `driftlens.py` | DriftLens core (PCA, Gaussian stats, FDD, calibration, save/load). Pure numpy/scipy/sklearn |
| `embedder.py` | Sentence Transformer wrapper with the per-window cache |
| `config.yaml` | All settings |
| `requirements.txt` | `sentence-transformers`, `scikit-learn`, `scipy`, `numpy`, `pyyaml`, `pytest` |
| `tests/test_driftlens.py` | 12 tests on synthetic data, no model download needed |

## How to run (from the repository root)

```bash
pip install -r driftlens/requirements.txt
python driftlens/run_driftlens.py --refit          # full batch run on windows_all.jsonl
```

| Option | Meaning |
|---|---|
| `--config PATH` | Config file (default `driftlens/config.yaml`) |
| `--windows-path PATH` | Override the input (JSONL file or Task 2's `output/windows` folder) |
| `--max-windows N` | Only the first N windows (quick test) |
| `--refit` | Ignore a saved reference and rebuild it |
| `--follow` | Keep polling for new windows (run alongside Task 2) |

**GPU.** `pip install sentence-transformers` pulls a CPU-only PyTorch on Windows. For an NVIDIA GPU, install the CUDA build that matches the driver (`nvidia-smi` shows the CUDA version), for example:

```bash
pip install "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```

On the development laptop (RTX 3050 6 GB) this raised embedding speed from about 170 to about 4,600 tweets/s. No code or config change is needed; the log line `Loaded Sentence Transformer ... on cuda:0` confirms the GPU is used.

**Tests:**

```bash
python -m pytest driftlens/tests -q
```

## Output

`output/drift/driftlens.jsonl`, one line per window:

```json
{"window_id": 18402, "window_start": "2020-05-20T00:00:00", "driftlens_score": 0.029159, "driftlens_alarm": true,
 "threshold": 0.028602, "n_posts": 1445, "n_used": 1000, "is_reference": false}
```

`n_posts` is the window's tweet count from Task 2, `n_used` the number scored (at most 1000), and `threshold` the one used for that size.

## Results on the full dataset

Input: 153 daily windows (`window_id` 18371-18804, 2020-04-19 to 2021-06-26, 410,455 tweets). There are no windows for July 2020 or November 2020 - March 2021.

**Reference period** (14 windows, 18371-18386): 49,637 tweets; the 150 PCA dimensions keep 83.5% of the variance; the threshold pool has 24,818 tweets. Threshold for full 1000-tweet windows: **0.0286**.

- The 12 reference windows with 896-1000 tweets score 0.0215-0.0291, all under their thresholds, as they should.
- The 2 reference windows that alarm are the two smallest in the dataset: 18371 (339 tweets, 0.0738 vs 0.0700) and 18375 (163 tweets, 0.1509 vs 0.1476). Both are only just over the threshold. With so few tweets the covariance estimate is noisy, and a p99 threshold allows about 1% false alarms anyway.

**After the reference period:** 122 of 139 windows raise an alarm.

| Month | Windows | Alarms | Median score |
|---|---|---|---|
| 2020-05 (after 05-04) | 26 | 10 | 0.0267 |
| 2020-06 | 17 | 16 | 0.0363 |
| 2020-08 | 10 | 10 | 0.0414 |
| 2020-09 | 19 | 19 | 0.0358 |
| 2020-10 | 20 | 20 | 0.0398 |
| 2021-04 | 5 | 5 | 0.0695 |
| 2021-05 | 24 | 24 | 0.0610 |
| 2021-06 | 18 | 18 | 0.0575 |

What this shows:

- **Gradual drift.** May 2020 stays close to the reference: the median is under the threshold and alarms are scattered. The first run of 3 consecutive alarms ends on 2020-05-30 (window 18412). From June 2020 on, nearly every window alarms.
- **A step up in 2021.** Median scores in April-June 2021 (0.058-0.070) are about twice those of mid-2020 (0.036-0.041). The conversation a year later is much further from April 2020 than the summer 2020 conversation was.
- **Highest scores:** 18537 (2020-10-02, 0.163), 18744 (2021-04-27, 0.144), 18792 (2021-06-14, 0.107). Several of the top windows are small (209-422 tweets) and have correspondingly higher thresholds. Among full 1000-tweet windows, the largest scores are 2021-06-14 (0.107) and 2020-10-03 / 10-05 (about 0.080). 2020-10-02 is the day the US president's COVID-19 diagnosis was announced, and the following days also score high.
- The single-day (grouped) calibration matters. A threshold calibrated on a random mix of all reference days was 0.0214, below several normal reference days (up to 0.0277). That would have made almost every window alarm, including the reference itself.

**Run time.** The full run took 31 minutes with embedding on the GPU. Embedding was a small part of that. Almost all of the time went to CPU threshold calibration: every window with a new size under 1000 tweets needs 1000 FDD computations (a 150x150 matrix square root each), about 80 s per distinct size. Calibrated thresholds are stored in `reference.npz`, so a re-run without `--refit` reuses them. Lowering `n_calibration` or `pca_components` makes this faster, at the cost of a noisier threshold or less of the variance kept.

## Design decisions

- **Reference = first 14 windows.** The start of the stream is taken as normal. 14 windows give about 50k tweets, enough for a stable 150-d covariance (the baseline half has about 25k rows for 11k covariance entries).
- **Fixed sample size of 1000.** FDD has a positive bias that shrinks as the sample grows, so windows of 3,000 and 300 tweets are not comparable. Sampling to 1000, with size-specific thresholds below that, keeps scores comparable and makes the per-window cost constant.
- **p99 threshold.** About 1 false alarm per 100 normal windows. In a stream of daily windows that is roughly one every three months, a reasonable rate for a monitoring signal.
- **No re-fitting on drift.** The reference is fixed. The task is to measure how far the conversation moves from its starting point, not to adapt to it; a sliding reference would hide the gradual drift shown above.
