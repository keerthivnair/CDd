# Experiment Scenarios: test cases for DriftLens and MCD-DD

Five reproducible test streams, built **only from our real COVID tweet windows** (`windows_all.jsonl`), for running and comparing both drift detectors. Each controlled scenario records exactly where the change was introduced, so detector alarms can be scored against a known answer.

| # | Scenario | What the stream looks like | Ground truth |
|---|---|---|---|
| 1 | `natural` | The 153 original windows, unchanged and in date order | Unknown (real data); first 14 windows are the reference |
| 2 | `sudden` | Old concept for windows 0–29, new concept for windows 30–59 | Drift at **window 30** |
| 3 | `gradual` | Old for windows 0–29. Windows 30–44 mix old and new, with the new share rising 6% → 94%. 100% new for windows 45–59 | Drift **starts at 30**, first fully-new window **45** |
| 4 | `volume` | Old concept throughout. 1000 tweets/window until window 29, then **3000** from window 30 on | **No drift** (volume change at 30) |
| 5 | `no_drift` | Old concept throughout, 1000 tweets/window | **No drift** |

Positions are 0-based. Windows 0–13 are the detectors' reference period in every scenario.

---

## How the scenarios are built

### Two concepts, both real

Instead of inventing text, a "concept" is a period of real COVID discourse:

| Concept | Source windows | Distinct tweets | What people were talking about |
|---|---|---|---|
| **old** | 2020-04-19 → 2020-05-31 (41 days) | 124,426 | First lockdowns, case counts, reopening |
| **new** | 2021-04-26 → 2021-06-26 (49 days) | 137,512 | Vaccination rollout, second-wave news |

The two periods are a year apart. The full natural run already shows the 2021 windows sitting at about 2× the reference Fréchet distance, so the two concepts really do differ in embedding space. The old period also contains the 14 reference windows the detectors are fitted on, so "old concept" means the same thing the detectors learned as normal.

### Window construction

Every controlled window is filled by **drawing real tweets at random, without replacement**, from the two pools:

- **Fixed window size.** Each window holds 1000 tweets, the same as DriftLens `window_sample_size`, so window size never changes by accident. The `volume` scenario is the only one where size changes, and there it is the variable under test.
- **Stationary concept.** Within a concept, every window is an i.i.d. sample of the same pool. The only change in a stream is the one we inject. Note that the natural stream itself drifts *inside* the old period: DriftLens alarms on 10 of the 26 post-reference May 2020 windows. Sampling from a pool is what keeps the `no_drift` and pre-change parts clean.
- **No reuse.** A tweet appears at most once per scenario, and exact duplicate texts such as retweets are dropped from the pools. Tweet order inside each window is shuffled, so mixed windows don't put all the old tweets first.
- **Synthetic dates.** Windows get consecutive synthetic dates starting 2000-01-01. `window_id` keeps the Task 2 convention (days since epoch), so it never collides with real window ids or the main embedding cache, and nobody mistakes the dates for real ones.
- **Same text filter as the detectors.** Tweets that become empty after `drift/embedder.py`'s U+FFFD clean-up are excluded, so `post_count` always equals the number of embeddings.

This follows the usual way of building drift benchmarks from real data: inject a controlled change at a known point and keep the real data's complexity. DriftLens itself is evaluated this way, using fixed-size windows built from a mix of drifted and non-drifted samples, with sudden drift after a fixed number of windows and incremental drift where the drifted share grows each window ([Greco et al., 2024](https://arxiv.org/abs/2406.17813)). The gradual scenario uses the common linear schedule, where the probability of drawing from the new distribution rises linearly from 0 at drift start to 1 at drift end ([benchmark framework, 2026](https://arxiv.org/abs/2606.07789)). The drift types follow the standard taxonomy of sudden, gradual, incremental and recurring drift. MCD-DD is evaluated on both synthetic and real-world drift streams ([Wan et al., 2024](https://arxiv.org/abs/2407.05375)).

### What each scenario tests

| Scenario | Question it answers |
|---|---|
| `natural` | What do the detectors say on the real stream? This is the reference behaviour, with no answer key. |
| `sudden` | Does the detector fire at the change point, and how many windows late? |
| `gradual` | How early in the transition does it notice? DriftLens is expected to fire part-way through. MCD-DD compares neighbouring windows that differ by only ~6%, so it may stay quiet. |
| `volume` | Does a traffic spike alone cause false alarms? DriftLens subsamples to 1000 so it should be immune. MCD-DD sees all tweets. |
| `no_drift` | Baseline false alarm rate when nothing changes. |

---

## Files

| Path | Purpose | In git? |
|---|---|---|
| `config.yaml` | Concept periods, stream length, window size, change point, seed | yes |
| `scenarios.py` | Builds all five scenarios | yes |
| `evaluate.py` | Scores detector alarms against ground truth → `results/summary.{json,md}` | yes |
| `tests/test_scenarios.py` | 13 tests on synthetic windows: composition, no reuse, reproducibility, ground truth | yes |
| `scenarios/index.json` | One-line summary of every scenario and its change point | yes |
| `scenarios/<name>/scenario.json` | **Ground truth**, plus per-window composition (`n_old`, `n_new`, `frac_new`, `phase`, source window ids) and the windows file's sha256 | yes |
| `scenarios/<name>/config.yaml` | Ready-to-use `drift/run_drift.py` config, with outputs under `output/experiments/<name>/` | yes |
| `scenarios/<name>/windows.jsonl` | The stream, in the same format as `windows_all.jsonl` | no (9–18 MB, regenerate in ~3 s) |
| `results/summary.md` | Latest detector results on all five scenarios | yes |

`scenario.json` ground-truth block (`sudden` shown):

```json
"ground_truth": {
  "concept_drift": true,
  "drift_start": {"position": 30, "window_id": 10987, "window_start": "2000-01-31T00:00:00"},
  "drift_end":   {"position": 30, "window_id": 10987, "window_start": "2000-01-31T00:00:00"},
  "volume_change": null,
  "expected_alarm_windows": "positions 30-59"
}
```

---

## How to run

From the repository root, with `windows_all.jsonl` present (`python drift/prepare_windows.py` if not):

```bash
# 1. Build the scenarios (deterministic: seed 42, same sha256 every time)
python experiments/scenarios.py

# 2. Run both detectors on a scenario (unchanged drift/run_drift.py, just a different config)
python drift/run_drift.py --config experiments/scenarios/sudden/config.yaml --refit

# 3. Score all scenarios that have been run against their ground truth
python experiments/evaluate.py

# Tests (no dataset or model needed)
pytest experiments/tests/ -v
```

Bash loop to run all five, about 30 minutes on a laptop GPU (each controlled scenario embeds 60k–120k tweets; `natural` reuses the main embedding cache but is slowed by threshold calibration, see Results):

```bash
for s in natural sudden gradual volume no_drift; do python drift/run_drift.py --config experiments/scenarios/$s/config.yaml --refit; done
```

Every scenario writes its scores, models and embedding cache to `output/experiments/<name>/`, so the main `output/drift/` results and the Grafana feed are untouched. To view a scenario in Grafana, point the monitor at it: `python -m monitoring.ingest --source output/experiments/sudden/scores.jsonl`.

**Changing a scenario:** edit `experiments/config.yaml` (for example `change_point`, `gradual_length`, `volume_factor` or `window_size`) and rerun step 1. The ground truth and configs are regenerated to match.

---

## Metrics reported by `evaluate.py`

Only windows after the 14 reference windows count.

| Metric | Meaning |
|---|---|
| `false_alarm_rate_before_drift` | Share of pre-change windows that alarmed (should be ~0) |
| `first_alarm_after_drift` / `detection_delay` | First alarm at or after drift start, and how many windows late |
| `detected_within_tolerance` | First alarm ≤ 3 windows after drift end (`--tolerance`) |
| `alarm_rate_after_drift` | Share of post-change windows that alarmed. DriftLens should stay high. MCD-DD compares consecutive windows, so a single alarm at the change is its expected pattern. |
| `false_alarm_rate` (no_drift), `…_before/after_volume_change` (volume) | Every alarm here is false |

## Results

Both detectors ran unchanged with the default `drift/config.yaml` settings (2026-10-03, RTX 3050 laptop GPU). The full table is in [`results/summary.md`](results/summary.md) and the per-window numbers are in `results/summary.json`.

| Scenario | Expected | DriftLens | MCD-DD |
|---|---|---|---|
| natural | unknown | 122/139 alarms (matches the main Task 3 run) | 2/139 alarms |
| sudden | drift at 30 | **0 false alarms; fires at 30 (delay 0)**; 100% of windows after the change | no alarms |
| gradual | drift 30→45 | **0 false alarms; fires at 32** (19% new tweets, delay 2); 93% of windows after the change | no alarms |
| volume | no drift | **0 alarms** (×3 volume ignored) | 0 alarms |
| no_drift | no drift | **0 alarms** | 0 alarms |

What the scenarios show:

- **They behave as designed.** The streams with no injected change (`no_drift` and `volume`) give zero alarms from both detectors. The sudden change shows up as a clean step in DriftLens's score, from about 0.022 to 0.050 against a threshold of 0.023. In the gradual scenario the score climbs steadily with the share of new tweets (0.021 at 30, 0.036 at 39, 0.046 at 45). So the controlled change, and nothing else, is what moves the detectors.
- **The volume case isolates volume.** DriftLens samples 1000 tweets from every window, so tripling the volume leaves its score flat: 0.0221 before, 0.0222 after.
- **MCD-DD sees the sudden change but does not alarm.** At window 30 its score is 8.3e-5, about 21× the pre-change median (4e-6) and 6× the pre-change maximum. Its dynamic threshold at that window is 1.7e-4, so no alarm fires. The threshold also grows during online training, from about 1.7e-4 at window 30 to about 6e-3 by the end. This is a finding for whoever tunes MCD-DD; the scenarios themselves are fine. The `gradual` case is harder for MCD-DD by design, because neighbouring windows differ by only ~6%.

### Why MCD-DD gives 0 alarms

The runs themselves are valid. In every scenario MCD-DD is trained from scratch (`--refit`) on the 14 reference windows and scores every window with non-zero scores, and its 10 unit tests pass. Replaying the `sudden` run separately reproduces the same numbers. The zeros come from how `drift/mcddd.py` sets its alarm threshold:

| `sudden` window | MCD-DD score | Threshold | Spread between two 100-tweet samples of one window (95th percentile) |
|---|---|---|---|
| 28 | 1.6e-6 | 3.4e-4 | 2.1e-4 |
| 29 | 1.3e-5 | 2.4e-4 | 3.0e-4 |
| **30 (drift)** | **8.3e-5** | 1.7e-4 | 2.5e-4 |
| 31 | 3.4e-6 | 2.9e-4 | 4.4e-4 |
| 59 | 6.7e-5 | 6.2e-3 | 9.3e-3 |

1. **Score and threshold use different sample sizes.** `_detect()` scores the encoder output of a whole 1000-tweet window against the whole previous window. The threshold (`_train_step()`) is the 95th percentile of distances between two 100-tweet samples taken *from the same window*. Mean-pooling over 100 items is much noisier than over 1000, so the threshold sits at small-sample noise level (last column ≈ threshold column). The drift at window 30 is ~20× the normal window-to-window score, but still below that threshold.
2. **The threshold inflates during the stream.** `score()` takes a training step on every incoming window. Each step pushes all encoder distances further apart, so the threshold grows from 1.4e-4 to 6.2e-3 (about 40×) while real changes stay around 1e-4.
3. **The encoder is barely trained.** It gets 3 steps at fit and 1 per window after that, and the "strong negative" pairs add Gaussian noise with σ = 0.1 to 384-d unit vectors (noise norm ≈ 2, larger than the vector itself). So the encoder mostly learns to spot added noise rather than topic change. Even at the 100-tweet scale, the change at window 30 (median cross-window distance 1.0e-4) stays inside the same-window noise.

Possible fixes for the MCD-DD owner, none applied here:
- Calibrate the threshold at the same set size used for scoring, or score 100-tweet sample sets, as the threshold does.
- Freeze or normalise the threshold after the offline fit instead of updating it every window.
- Train for more steps, and reconsider `eps_big`.

Rerun the five scenario configs after any change to compare.

Runtime: each controlled scenario takes about 2–4 minutes. `natural` takes about 18 minutes, because DriftLens calibrates a new threshold (about 30 s) for every distinct window size under 1000.
