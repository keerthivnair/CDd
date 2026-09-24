"""Task 3: cleaned tweet windows -> Sentence Transformer embeddings -> DriftLens drift score + alarm.

Run from the repository root:
    python driftlens/run_driftlens.py --config driftlens/config.yaml
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driftlens import DriftLens  # noqa: E402

log = logging.getLogger("driftlens")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_windows(path: str) -> list:
    """Windows sorted by window_id, from one JSONL file or Task 2's batch_id=*/part-*.json folder."""
    files = [path] if os.path.isfile(path) else glob.glob(os.path.join(path, "batch_id=*", "part-*.json"))
    by_id = {}
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    w = json.loads(line)
                    by_id[w["window_id"]] = w
    return [by_id[k] for k in sorted(by_id)]


def reference_meta(cfg: dict, ref_ids: list) -> dict:
    return {"model": cfg["embedding"]["model"], "reference_ids": np.array(ref_ids, dtype=np.int64)}


def get_detector(cfg: dict, embedder, windows: list, refit: bool) -> tuple:
    d = cfg["driftlens"]
    ref = windows[: d["reference_windows"]]
    ref_ids = [w["window_id"] for w in ref]
    path = resolve(cfg["output"]["reference_path"])
    params = [d["pca_components"], d["threshold_fraction"], d["n_calibration"],
              d["percentile"], d["window_sample_size"], d["seed"]]

    if os.path.exists(path) and not refit:
        dl, meta = DriftLens.load(path)
        same = (str(meta.get("model")) == cfg["embedding"]["model"]
                and meta.get("reference_ids", np.array([])).tolist() == ref_ids
                and [dl.pca_components, dl.threshold_fraction, dl.n_calibration,
                     dl.percentile, dl.window_sample_size, dl.seed] == params)
        if same:
            log.info("Loaded reference from %s", path)
            return dl, set(ref_ids)
        log.info("Saved reference does not match the config; refitting")

    log.info("Building reference from %s windows (ids %s-%s)", len(ref), ref_ids[0], ref_ids[-1])
    ref_parts = [embedder.embed_window(w) for w in ref]
    groups = np.concatenate([np.full(len(e), w["window_id"]) for e, w in zip(ref_parts, ref)])
    dl = DriftLens(d["pca_components"], d["threshold_fraction"], d["n_calibration"],
                   d["percentile"], d["window_sample_size"], d["seed"]).fit(np.vstack(ref_parts), groups)
    log.info("Reference: %s tweets, PCA %s dims (%.1f%% variance), threshold pool %s tweets",
             len(groups), dl.pca_basis.shape[0], 100 * dl.explained_variance, len(dl.threshold_pool))
    thr = dl.calibrate(d["window_sample_size"])
    log.info("Calibrated threshold for %s-tweet windows: %.4f (p%s of %s single-day windows)",
             d["window_sample_size"], thr, d["percentile"], d["n_calibration"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dl.save(path, **reference_meta(cfg, ref_ids))
    return dl, set(ref_ids)


def score_window(dl: DriftLens, embedder, window: dict, ref_ids: set) -> dict:
    t0 = time.perf_counter()
    emb = embedder.embed_window(window)
    r = dl.score(emb, window_id=window["window_id"])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    n_posts = window.get("post_count", len(window.get("texts", [])))
    throughput = (n_posts / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    return {
        "window_id": window["window_id"],
        "window_start": window.get("window_start"),
        "driftlens_score": round(r["driftlens_score"], 6),
        "driftlens_alarm": r["driftlens_alarm"],
        "threshold": round(r["threshold"], 6),
        "n_posts": n_posts,
        "n_used": r["n_used"],
        "is_reference": window["window_id"] in ref_ids,
        "processing_latency_ms": round(latency_ms, 2),
        "throughput_posts_per_sec": round(throughput, 2),
    }


def log_result(res: dict) -> None:
    log.info("window=%s score=%.4f threshold=%.4f alarm=%s n=%s latency=%.1fms throughput=%.1fp/s%s",
             res["window_id"], res["driftlens_score"], res["threshold"], res["driftlens_alarm"],
             res["n_used"], res.get("processing_latency_ms", 0.0), res.get("throughput_posts_per_sec", 0.0),
             " (reference)" if res["is_reference"] else "")


def main():
    parser = argparse.ArgumentParser(description="Task 3: Embedding + DriftLens drift detection")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--windows-path", help="Override input.windows_path (JSONL file or output/windows folder)")
    parser.add_argument("--max-windows", type=int, help="Only process the first N windows (quick test)")
    parser.add_argument("--refit", action="store_true", help="Ignore a saved reference and rebuild it")
    parser.add_argument("--follow", action="store_true", help="Keep polling for new windows (streaming mode)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.windows_path:
        cfg["input"]["windows_path"] = args.windows_path
    windows_path = resolve(cfg["input"]["windows_path"])
    out_path = resolve(cfg["output"]["scores_path"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_ref = cfg["driftlens"]["reference_windows"]

    from embedder import Embedder
    e = cfg["embedding"]
    embedder = Embedder(e["model"], resolve(e["cache_dir"]) if e.get("cache_dir") else "",
                        e.get("batch_size", 256), e.get("device", ""))

    windows = load_windows(windows_path)[: args.max_windows]
    while args.follow and len(windows) < n_ref:
        log.info("Waiting for %s reference windows (have %s)", n_ref, len(windows))
        time.sleep(cfg["follow"]["poll_seconds"])
        windows = load_windows(windows_path)
    if len(windows) < n_ref:
        sys.exit(f"Need at least {n_ref} windows for the reference period, found {len(windows)} in {windows_path}")
    log.info("Loaded %s windows from %s", len(windows), windows_path)

    dl, ref_ids = get_detector(cfg, embedder, windows, args.refit)

    done, alarms = set(), 0
    with open(out_path, "w", encoding="utf-8") as out:
        while True:
            for w in windows:
                if w["window_id"] in done:
                    continue
                res = score_window(dl, embedder, w, ref_ids)
                out.write(json.dumps(res) + "\n")
                out.flush()
                done.add(w["window_id"])
                alarms += res["driftlens_alarm"] and not res["is_reference"]
                log_result(res)
            if not args.follow:
                break
            time.sleep(cfg["follow"]["poll_seconds"])
            windows = load_windows(windows_path)

    log.info("Done: %s windows scored -> %s", len(done), out_path)
    log.info("Alarms: %s of %s non-reference windows", alarms, len(done - ref_ids))
    dl.save(resolve(cfg["output"]["reference_path"]), **reference_meta(cfg, sorted(ref_ids)))


if __name__ == "__main__":
    main()
