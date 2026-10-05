"""Task 3: cleaned tweet windows -> Sentence Transformer embeddings -> DriftLens + MCD-DD drift scores + alarms.

Run from the repository root:
    python drift/run_drift.py --config drift/config.yaml
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
from mcddd import MCDDD  # noqa: E402

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


MCDDD_KEYS = ["hidden_dim", "output_dim", "n_encoders", "lr", "lambda_gp", "lipschitz_k", "eps_small", "eps_big",
              "n_sub_windows", "sample_set_size", "k_pairs", "train_epochs", "scoring_sample_size",
              "context_windows", "n_calibration", "percentile", "online_training", "holdout_fraction", "seed"]


def get_mcddd(cfg: dict, embedder, windows: list, ref_ids: set, refit: bool) -> MCDDD:
    """Build or load the MCD-DD detector, trained on the same reference embeddings as DriftLens."""
    m = cfg.get("mcddd", {})
    if not m.get("enabled", True):
        return None

    mcddd_path = resolve(m.get("model_path", "output/drift/mcddd.pt"))
    params = {k: m[k] for k in MCDDD_KEYS if k in m}  # keys missing from the config use MCDDD defaults
    device = m.get("device", cfg["embedding"].get("device", ""))
    ref_list = sorted(ref_ids)

    if os.path.exists(mcddd_path) and not refit:
        try:
            mcddd = MCDDD.load(mcddd_path, device=device)
            saved = mcddd.params()
            same = (all(saved[k] == v for k, v in params.items())
                    and list(mcddd.meta.get("reference_ids", [])) == ref_list
                    and mcddd.meta.get("model") == cfg["embedding"]["model"])
            if same:
                log.info("Loaded MCD-DD model from %s", mcddd_path)
                return mcddd
            log.info("Saved MCD-DD model does not match the config; refitting")
        except Exception as ex:
            log.warning("Failed to load MCD-DD from %s (%s); refitting", mcddd_path, ex)

    # Reuse the exact same cached embeddings as DriftLens; each reference window is one sub-window
    ref_windows = [w for w in windows if w["window_id"] in ref_ids]
    log.info("Fitting MCD-DD on %s reference windows (same embeddings as DriftLens)", len(ref_windows))
    ref_parts = [embedder.embed_window(w) for w in ref_windows]
    groups = np.concatenate([np.full(len(e), w["window_id"]) for e, w in zip(ref_parts, ref_windows)])
    t0 = time.perf_counter()
    mcddd = MCDDD(input_dim=ref_parts[0].shape[1], device=device, **params).fit(np.vstack(ref_parts), groups)
    log.info("MCD-DD trained + calibrated in %.1fs", time.perf_counter() - t0)

    os.makedirs(os.path.dirname(mcddd_path), exist_ok=True)
    mcddd.save(mcddd_path, reference_ids=ref_list, model=cfg["embedding"]["model"])
    log.info("MCD-DD fitted and saved to %s", mcddd_path)
    return mcddd


def sig(x, digits: int = 6):
    """Round to significant digits (fixed decimals would flatten very small scores to 0)."""
    return None if x is None else float(f"{x:.{digits}g}")


def score_window(dl: DriftLens, mcddd: MCDDD | None, embedder, window: dict, ref_ids: set) -> dict:
    t0 = time.perf_counter()

    # Embed once — both detectors use the exact same embeddings
    emb = embedder.embed_window(window)

    # DriftLens score
    r = dl.score(emb, window_id=window["window_id"])

    # MCD-DD score (same embeddings). Reference windows are its training data, so they are not scored;
    # fit() already placed their held-out rows in MCD-DD's sliding context.
    if mcddd is not None and window["window_id"] not in ref_ids:
        mr = mcddd.score(emb, window_id=window["window_id"])
    else:
        mr = {"mcddd_score": None, "mcddd_alarm": None, "mcddd_threshold": None, "mcddd_lag": None}

    latency_ms = (time.perf_counter() - t0) * 1000.0
    n_posts = window.get("post_count", len(window.get("texts", [])))
    throughput = (n_posts / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    result = {
        "window_id": window["window_id"],
        "window_start": window.get("window_start"),
        # DriftLens outputs
        "driftlens_score": round(r["driftlens_score"], 6),
        "driftlens_alarm": r["driftlens_alarm"],
        "threshold": round(r["threshold"], 6),
        # MCD-DD outputs
        "mcddd_score": sig(mr["mcddd_score"]),
        "mcddd_alarm": mr["mcddd_alarm"],
        "mcddd_threshold": sig(mr.get("mcddd_threshold")),
        "mcddd_lag": mr.get("mcddd_lag"),
        # Metadata
        "n_posts": n_posts,
        "n_used": r["n_used"],
        "is_reference": window["window_id"] in ref_ids,
        "processing_latency_ms": round(latency_ms, 2),
        "throughput_posts_per_sec": round(throughput, 2),
    }
    return result


def log_result(res: dict) -> None:
    mcd_part = ""
    if res.get("mcddd_score") is not None:
        val = res["mcddd_score"]
        mcd_part = f" mcddd={val:.4g} mcddd_thr={res['mcddd_threshold']:.4g} mcddd_alarm={res['mcddd_alarm']}"
    log.info("window=%s driftlens=%.4f threshold=%.4f dl_alarm=%s%s n=%s latency=%.1fms throughput=%.1fp/s%s",
             res["window_id"], res["driftlens_score"], res["threshold"], res["driftlens_alarm"],
             mcd_part,
             res["n_used"], res.get("processing_latency_ms", 0.0), res.get("throughput_posts_per_sec", 0.0),
             " (reference)" if res["is_reference"] else "")


def main():
    parser = argparse.ArgumentParser(description="Task 3: Embedding + DriftLens + MCD-DD drift detection")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--windows-path", help="Override input.windows_path (JSONL file or output/windows folder)")
    parser.add_argument("--max-windows", type=int, help="Only process the first N windows (quick test)")
    parser.add_argument("--refit", action="store_true", help="Ignore a saved reference and rebuild it")
    parser.add_argument("--follow", action="store_true", help="Keep polling for new windows (streaming mode)")
    parser.add_argument("--no-mcddd", action="store_true", help="Disable MCD-DD (run DriftLens only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.windows_path:
        cfg["input"]["windows_path"] = args.windows_path
    if args.no_mcddd:
        cfg.setdefault("mcddd", {})["enabled"] = False
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

    # --- MCD-DD: trained on the same reference embeddings ---
    mcddd = get_mcddd(cfg, embedder, windows, ref_ids, args.refit)

    done, dl_alarms, mcd_alarms = set(), 0, 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        while True:
            for w in windows:
                if w["window_id"] in done:
                    continue
                res = score_window(dl, mcddd, embedder, w, ref_ids)
                out.write(json.dumps(res) + "\n")
                out.flush()
                done.add(w["window_id"])
                if not res["is_reference"]:
                    dl_alarms += res["driftlens_alarm"]
                    if res.get("mcddd_alarm"):
                        mcd_alarms += 1
                log_result(res)
            if not args.follow:
                break
            time.sleep(cfg["follow"]["poll_seconds"])
            windows = load_windows(windows_path)

    n_non_ref = len(done - ref_ids)
    log.info("Done: %s windows scored -> %s", len(done), out_path)
    log.info("DriftLens alarms: %s of %s non-reference windows", dl_alarms, n_non_ref)
    if mcddd is not None:
        log.info("MCD-DD alarms:    %s of %s non-reference windows", mcd_alarms, n_non_ref)
    dl.save(resolve(cfg["output"]["reference_path"]), **reference_meta(cfg, sorted(ref_ids)))


if __name__ == "__main__":
    main()
