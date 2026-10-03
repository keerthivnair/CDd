"""Build the five evaluation scenarios from the real COVID tweet windows.

  natural  - the original windows, unchanged (no injected change; real drift unknown)
  sudden   - old concept, then new concept from the change point on
  gradual  - old concept, then the share of new-concept tweets rises linearly to 100%
  volume   - old concept throughout, tweets per window multiplied from the change point on
  no_drift - old concept throughout

Controlled scenarios draw tweets at random without replacement from two pools of real tweets
(old = early 2020, new = mid 2021), so the only change in the stream is the one we inject.
Each scenario writes scenario.json (ground truth, per-window composition), config.yaml (ready
for drift/run_drift.py) and windows.jsonl (same format as windows_all.jsonl).

Run from the repository root:
    python experiments/scenarios.py
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "drift"))
from embedder import prepare_texts  # noqa: E402  (same text filter the detectors apply)

log = logging.getLogger("experiments")
ISO = "%Y-%m-%dT%H:%M:%S"
CONTROLLED = ["sudden", "gradual", "volume", "no_drift"]
SCENARIO_SEEDS = {"sudden": 1, "gradual": 2, "volume": 3, "no_drift": 4}

DESCRIPTIONS = {
    "natural": "Original COVID windows in chronological order, unchanged. No injected change; "
               "the true drift points are unknown.",
    "sudden": "Old concept (early 2020) up to the change point, then the new concept (mid 2021) "
              "from the change point on, with no transition.",
    "gradual": "Old concept up to the change point, then each window mixes old and new tweets "
               "with the new share rising linearly over the transition, then 100% new.",
    "volume": "Old concept throughout. From the change point on each window has volume_factor "
              "times more tweets. The concept does not change.",
    "no_drift": "Old concept throughout at a constant window size. Nothing changes.",
}


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_windows(path: str) -> list:
    """Windows from a JSONL file, sorted by window_id."""
    with open(path, encoding="utf-8") as f:
        windows = [json.loads(line) for line in f if line.strip()]
    return sorted(windows, key=lambda w: w["window_id"])


def build_pool(windows: list, start: str, end: str) -> list:
    """(text, post_id, source_window_id) for every distinct tweet text in windows starting in [start, end]."""
    pool, seen = [], set()
    for w in windows:
        if not start <= w["window_start"][:10] <= end:
            continue
        for text, pid in zip(w["texts"], w["post_ids"]):
            kept = prepare_texts([text])
            if kept and kept[0] not in seen:
                seen.add(kept[0])
                pool.append((kept[0], pid, w["window_id"]))
    return pool


class Sampler:
    """Draws tweets from a pool at random without replacement."""

    def __init__(self, pool: list, rng: np.random.Generator, name: str):
        self.pool, self.order, self.pos, self.name = pool, rng.permutation(len(pool)), 0, name

    def take(self, n: int) -> list:
        if self.pos + n > len(self.pool):
            raise ValueError(f"{self.name} pool exhausted: need {self.pos + n} tweets, have {len(self.pool)}")
        idx = self.order[self.pos:self.pos + n]
        self.pos += n
        return [self.pool[i] for i in idx]


def plan(name: str, s: dict) -> list:
    """(n_old, n_new) tweets for each window position of a controlled scenario."""
    n, size, cp = s["n_windows"], s["window_size"], s["change_point"]
    rows = []
    for p in range(n):
        if name == "sudden":
            rows.append((size, 0) if p < cp else (0, size))
        elif name == "gradual":
            frac = min(max(p - cp + 1, 0) / (s["gradual_length"] + 1), 1.0)
            n_new = int(round(frac * size))
            rows.append((size - n_new, n_new))
        elif name == "volume":
            rows.append((size if p < cp else size * s["volume_factor"], 0))
        elif name == "no_drift":
            rows.append((size, 0))
        else:
            raise ValueError(f"unknown scenario {name}")
    return rows


def phase_of(name: str, p: int, s: dict) -> str:
    cp = s["change_point"]
    if p < s["reference_windows"]:
        return "reference"
    if name == "no_drift" or p < cp:
        return "before_change"
    if name == "gradual" and p < cp + s["gradual_length"]:
        return "transition"
    return "after_change"


def ground_truth(name: str, s: dict, start: date) -> dict:
    def at(p):
        d = start + timedelta(days=p)
        return {"position": p, "window_id": (d - date(1970, 1, 1)).days,
                "window_start": d.strftime("%Y-%m-%dT00:00:00")}

    cp = s["change_point"]
    if name == "sudden":
        return {"concept_drift": True, "drift_start": at(cp), "drift_end": at(cp),
                "volume_change": None, "expected_alarm_windows": f"positions {cp}-{s['n_windows'] - 1}"}
    if name == "gradual":
        end = cp + s["gradual_length"]
        return {"concept_drift": True, "drift_start": at(cp), "drift_end": at(end),
                "volume_change": None, "expected_alarm_windows": f"positions {cp}-{s['n_windows'] - 1}",
                "note": f"positions {cp}-{end - 1} mix old/new tweets; position {end} is the first 100% new window"}
    if name == "volume":
        return {"concept_drift": False, "drift_start": None, "drift_end": None,
                "volume_change": {**at(cp), "factor": s["volume_factor"]}, "expected_alarm_windows": "none"}
    return {"concept_drift": False, "drift_start": None, "drift_end": None,
            "volume_change": None, "expected_alarm_windows": "none"}


def build_controlled(name: str, pools: dict, s: dict, seed: int) -> tuple:
    rng = np.random.default_rng([seed, SCENARIO_SEEDS[name]])
    old, new = Sampler(pools["old"], rng, "old"), Sampler(pools["new"], rng, "new")
    start = date.fromisoformat(s["start_date"])
    windows, rows = [], []
    for p, (n_old, n_new) in enumerate(plan(name, s)):
        items = old.take(n_old) + new.take(n_new)
        items = [items[i] for i in rng.permutation(len(items))]
        d = datetime.combine(start + timedelta(days=p), datetime.min.time(), tzinfo=timezone.utc)
        wid = int(d.timestamp() // 86400)
        phase = phase_of(name, p, s)
        windows.append({
            "window_id": wid,
            "window_start": d.strftime(ISO),
            "window_end": (d + timedelta(days=1)).strftime(ISO),
            "texts": [t for t, _, _ in items],
            "timestamps": [d.strftime(ISO)] * len(items),
            "post_ids": [pid for _, pid, _ in items],
            "post_count": len(items),
            "scenario": name,
            "position": p,
            "phase": phase,
        })
        rows.append({"position": p, "window_id": wid, "window_start": d.strftime(ISO), "phase": phase,
                     "n_posts": len(items), "n_old": n_old, "n_new": n_new,
                     "frac_new": round(n_new / len(items), 4),
                     "source_window_ids": sorted({src for _, _, src in items})})
    return windows, rows, ground_truth(name, s, start), {"old": old.pos, "new": new.pos}


def scenario_config(base: dict, name: str, windows_path: str, results_dir: str, cache_dir: str) -> dict:
    cfg = copy.deepcopy(base)
    out = f"{results_dir}/{name}"
    cfg["input"]["windows_path"] = windows_path
    cfg["embedding"]["cache_dir"] = cache_dir
    cfg["mcddd"]["model_path"] = f"{out}/mcddd.pt"
    cfg["output"]["reference_path"] = f"{out}/reference.npz"
    cfg["output"]["scores_path"] = f"{out}/scores.jsonl"
    return cfg


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def write_yaml(path: str, name: str, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-generated by experiments/scenarios.py for scenario '{name}'. Do not edit by hand.\n"
                f"# Run: python drift/run_drift.py --config {os.path.relpath(path, PROJECT_ROOT).replace(os.sep, '/')} --refit\n")
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Build the drift-detector evaluation scenarios")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(resolve(cfg["base_drift_config"])) as f:
        base = yaml.safe_load(f)
    s, seed = cfg["stream"], cfg["seed"]
    if s["reference_windows"] != base["driftlens"]["reference_windows"]:
        sys.exit("stream.reference_windows must equal driftlens.reference_windows in the drift config")
    if not s["reference_windows"] < s["change_point"] < s["n_windows"] - s.get("gradual_length", 0):
        sys.exit("need reference_windows < change_point < n_windows - gradual_length")

    windows = load_windows(resolve(cfg["source_windows"]))
    log.info("Loaded %s real windows (%s tweets) from %s", len(windows),
             sum(w["post_count"] for w in windows), cfg["source_windows"])
    pools = {k: build_pool(windows, v["start"], v["end"]) for k, v in cfg["concepts"].items()}
    for k, v in cfg["concepts"].items():
        log.info("Concept '%s' pool: %s distinct tweets from windows %s to %s", k, len(pools[k]), v["start"], v["end"])

    root, results = resolve(cfg["scenarios_dir"]), cfg["results_dir"]
    rel_root = cfg["scenarios_dir"]
    index = []

    # 1. Natural stream: the real windows, unchanged. Reuses the main embedding cache (same window ids).
    os.makedirs(os.path.join(root, "natural"), exist_ok=True)
    write_yaml(os.path.join(root, "natural", "config.yaml"), "natural",
               scenario_config(base, "natural", cfg["source_windows"], results, base["embedding"]["cache_dir"]))
    nat = {
        "scenario": "natural", "description": DESCRIPTIONS["natural"],
        "windows_path": cfg["source_windows"], "windows_sha256": sha256(resolve(cfg["source_windows"])),
        "n_windows": len(windows), "n_posts": sum(w["post_count"] for w in windows),
        "ground_truth": {"concept_drift": None, "drift_start": None, "drift_end": None, "volume_change": None,
                         "expected_alarm_windows": "unknown (real data)",
                         "note": "first 14 windows are the detector reference period"},
        "windows": [{"position": p, "window_id": w["window_id"], "window_start": w["window_start"],
                     "phase": "reference" if p < s["reference_windows"] else "stream", "n_posts": w["post_count"]}
                    for p, w in enumerate(windows)],
    }
    write_json(os.path.join(root, "natural", "scenario.json"), nat)
    index.append({"scenario": "natural", "n_windows": nat["n_windows"], "n_posts": nat["n_posts"],
                  "concept_drift": None, "drift_start_position": None, "drift_end_position": None,
                  "volume_change_position": None})

    # 2-5. Controlled scenarios.
    for name in CONTROLLED:
        out_dir = os.path.join(root, name)
        os.makedirs(out_dir, exist_ok=True)
        wins, rows, truth, used = build_controlled(name, pools, s, seed)
        wpath = os.path.join(out_dir, "windows.jsonl")
        with open(wpath, "w", encoding="utf-8", newline="\n") as f:  # "\n" keeps the sha256 OS-independent
            for w in wins:
                f.write(json.dumps(w) + "\n")
        rel_windows = f"{rel_root}/{name}/windows.jsonl"
        write_yaml(os.path.join(out_dir, "config.yaml"), name,
                   scenario_config(base, name, rel_windows, results, f"{results}/{name}/embeddings"))
        write_json(os.path.join(out_dir, "scenario.json"), {
            "scenario": name, "description": DESCRIPTIONS[name],
            "windows_path": rel_windows, "windows_sha256": sha256(wpath),
            "source_windows": cfg["source_windows"], "concepts": cfg["concepts"],
            "pool_sizes": {k: len(v) for k, v in pools.items()}, "tweets_used": used,
            "seed": [seed, SCENARIO_SEEDS[name]], "stream": s,
            "n_windows": len(wins), "n_posts": sum(r["n_posts"] for r in rows),
            "ground_truth": truth, "windows": rows,
        })
        index.append({"scenario": name, "n_windows": len(wins), "n_posts": sum(r["n_posts"] for r in rows),
                      "concept_drift": truth["concept_drift"],
                      "drift_start_position": truth["drift_start"]["position"] if truth["drift_start"] else None,
                      "drift_end_position": truth["drift_end"]["position"] if truth["drift_end"] else None,
                      "volume_change_position": truth["volume_change"]["position"] if truth["volume_change"] else None})
        log.info("Scenario %-8s %s windows, %s tweets (old %s, new %s) -> %s", name, len(wins),
                 sum(r["n_posts"] for r in rows), used["old"], used["new"], rel_windows)

    write_json(os.path.join(root, "index.json"), {"seed": seed, "stream": s, "concepts": cfg["concepts"],
                                                  "scenarios": index})
    log.info("Wrote scenario index -> %s/index.json", rel_root)


if __name__ == "__main__":
    main()
