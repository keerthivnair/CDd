"""MCD-DD robustness: rerun only MCD-DD on every controlled scenario for several model seeds.

MCD-DD's encoder is trained from a random start, so one run can be lucky or unlucky. This script
reuses the embeddings cached by
    python drift/run_drift.py --config experiments/scenarios/<name>/config.yaml --refit
(run that first), refits MCD-DD with each seed in --seeds using the scenario's own config, scores
the stream, and scores the alarms with evaluate.py. Writes experiments/results/mcddd_seeds.{json,md}.

Run from the repository root:
    python experiments/mcddd_seeds.py --seeds 42 1 2 3 4
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "drift"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate, resolve  # noqa: E402
from mcddd import MCDDD  # noqa: E402
from run_drift import MCDDD_KEYS, load_windows  # noqa: E402

CONTROLLED = ["sudden", "gradual", "volume", "no_drift"]
log = logging.getLogger("experiments.mcddd_seeds")


def run_scenario(name: str, seeds: list, embedder_cls, tolerance: int, overrides: dict) -> list:
    sdir = resolve(os.path.join("experiments/scenarios", name))
    with open(os.path.join(sdir, "scenario.json")) as f:
        truth = json.load(f)["ground_truth"]
    with open(os.path.join(sdir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    e, m, n_ref = cfg["embedding"], cfg["mcddd"], cfg["driftlens"]["reference_windows"]
    embedder = embedder_cls(e["model"], resolve(e["cache_dir"]), e.get("batch_size", 256), e.get("device", ""))
    windows = load_windows(resolve(cfg["input"]["windows_path"]))
    emb = [embedder.embed_window(w) for w in windows]  # cache hits after a run_drift.py run
    groups = np.concatenate([np.full(len(x), w["window_id"]) for x, w in zip(emb[:n_ref], windows[:n_ref])])
    ref = np.vstack(emb[:n_ref])

    out = []
    for seed in seeds:
        params = {**{k: m[k] for k in MCDDD_KEYS if k in m}, **overrides, "seed": seed}
        det = MCDDD(device=m.get("device", ""), **params).fit(ref, groups)
        rows = [{"mcddd_alarm": False}] * n_ref + [
            det.score(x, window_id=w["window_id"]) for x, w in zip(emb[n_ref:], windows[n_ref:])]
        met = evaluate(truth, rows, n_ref, tolerance)["mcddd"]
        out.append({"seed": seed, **met})
        log.info("%-8s seed %-3s alarms %s/%s first %s", name, seed, met["alarms"], met["stream_windows"],
                 met.get("first_alarm_after_drift"))
    return out


def to_markdown(res: dict, seeds: list, overrides: dict) -> str:
    lines = [f"# MCD-DD across model seeds {seeds}", "",
             f"Scenario config MCD-DD settings{' with overrides ' + json.dumps(overrides) if overrides else ''}.", "",
             "Each cell: median over seeds [min-max]. Only MCD-DD is refit; embeddings and scenarios are fixed.", "",
             "| Scenario | Stream alarms | False alarms before change | First-alarm delay | Detected (runs) |",
             "|---|---|---|---|---|"]

    def rng(v):
        v = [x for x in v if x is not None]
        return "-" if not v else f"{np.median(v):g} [{min(v)}-{max(v)}]"

    for name, runs in res.items():
        alarms = rng([r["alarms"] for r in runs])
        if "false_alarms_before_drift" in runs[0]:
            fa = rng([r["false_alarms_before_drift"] for r in runs])
            delay = rng([r["detection_delay"] for r in runs])
            det = f"{sum(r['first_alarm_after_drift'] is not None for r in runs)}/{len(runs)}"
        else:
            fa, delay, det = alarms + " (all false)", "-", "-"
        lines.append(f"| {name} | {alarms} | {fa} | {delay} | {det} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="MCD-DD results across model seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--params", default="{}", help='JSON overrides of the mcddd config, e.g. \'{"train_epochs": 0}\'')
    parser.add_argument("--tag", default="", help="suffix for the output files (mcddd_seeds<tag>.md)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from embedder import Embedder

    overrides = json.loads(args.params)
    res = {name: run_scenario(name, args.seeds, Embedder, args.tolerance, overrides) for name in CONTROLLED}
    out_dir = resolve("experiments/results")
    with open(os.path.join(out_dir, f"mcddd_seeds{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": args.seeds, "overrides": overrides, "scenarios": res}, f, indent=2)
        f.write("\n")
    md = to_markdown(res, args.seeds, overrides)
    with open(os.path.join(out_dir, f"mcddd_seeds{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
