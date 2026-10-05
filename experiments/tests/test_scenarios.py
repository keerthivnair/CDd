"""Scenario generator + evaluator tests on small synthetic windows (no dataset or model needed).

Run from the repository root:
    pytest experiments/tests/ -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import evaluate  # noqa: E402
from scenarios import build_controlled, build_pool, plan  # noqa: E402

STREAM = {"n_windows": 20, "reference_windows": 5, "window_size": 10, "change_point": 8,
          "gradual_length": 4, "volume_factor": 3, "start_date": "2000-01-01"}


def fake_windows():
    """Old-period windows (texts 'old_*') in 2020 and new-period windows ('new_*') in 2021."""
    out = []
    for i, (day, tag) in enumerate([("2020-04-20", "old"), ("2020-04-21", "old"),
                                    ("2021-05-01", "new"), ("2021-05-02", "new")]):
        texts = [f"{tag}_{i}_{j}" for j in range(400)] + [f"{tag}_dup"]  # one duplicate per pool
        out.append({"window_id": 18000 + i, "window_start": f"{day}T00:00:00", "texts": texts,
                    "post_ids": [f"p{i}_{j}" for j in range(len(texts))]})
    return out


@pytest.fixture
def pools():
    w = fake_windows()
    return {"old": build_pool(w, "2020-04-19", "2020-05-31"), "new": build_pool(w, "2021-04-26", "2021-06-26")}


def test_pools_split_by_date_and_dedupe(pools):
    assert len(pools["old"]) == 801 and len(pools["new"]) == 801
    assert all(t.startswith("old") for t, _, _ in pools["old"])
    assert all(t.startswith("new") for t, _, _ in pools["new"])


@pytest.mark.parametrize("name", ["sudden", "gradual", "volume", "no_drift"])
def test_composition_matches_plan(pools, name):
    wins, rows, _, _ = build_controlled(name, pools, STREAM, seed=0)
    assert len(wins) == STREAM["n_windows"]
    for w, (n_old, n_new) in zip(wins, plan(name, STREAM)):
        assert sum(t.startswith("old") for t in w["texts"]) == n_old
        assert sum(t.startswith("new") for t in w["texts"]) == n_new
        assert w["post_count"] == len(w["texts"]) == len(w["post_ids"]) == len(w["timestamps"])


def test_no_tweet_reused_within_scenario(pools):
    for name in ["sudden", "gradual", "volume", "no_drift"]:
        wins, _, _, _ = build_controlled(name, pools, STREAM, seed=0)
        texts = [t for w in wins for t in w["texts"]]
        assert len(texts) == len(set(texts))


def test_reproducible(pools):
    a, _, _, _ = build_controlled("gradual", pools, STREAM, seed=7)
    b, _, _, _ = build_controlled("gradual", pools, STREAM, seed=7)
    c, _, _, _ = build_controlled("gradual", pools, STREAM, seed=8)
    assert a == b and a != c


def test_sudden_ground_truth(pools):
    wins, rows, truth, _ = build_controlled("sudden", pools, STREAM, seed=0)
    cp = STREAM["change_point"]
    assert truth["drift_start"]["position"] == truth["drift_end"]["position"] == cp
    assert truth["drift_start"]["window_id"] == wins[cp]["window_id"]
    assert all(r["frac_new"] == 0 for r in rows[:cp]) and all(r["frac_new"] == 1 for r in rows[cp:])
    assert [r["phase"] for r in rows[:STREAM["reference_windows"]]] == ["reference"] * STREAM["reference_windows"]


def test_gradual_is_monotonic(pools):
    _, rows, truth, _ = build_controlled("gradual", pools, STREAM, seed=0)
    cp, end = truth["drift_start"]["position"], truth["drift_end"]["position"]
    fr = [r["frac_new"] for r in rows]
    assert end == cp + STREAM["gradual_length"]
    assert all(f == 0 for f in fr[:cp]) and all(f == 1 for f in fr[end:])
    assert all(0 < f < 1 for f in fr[cp:end]) and np.all(np.diff(fr) >= 0)


def test_volume_changes_size_not_concept(pools):
    _, rows, truth, _ = build_controlled("volume", pools, STREAM, seed=0)
    cp, size = STREAM["change_point"], STREAM["window_size"]
    assert truth["concept_drift"] is False and truth["volume_change"]["position"] == cp
    assert [r["n_posts"] for r in rows] == [size] * cp + [size * 3] * (STREAM["n_windows"] - cp)
    assert all(r["n_new"] == 0 for r in rows)


def test_window_ids_are_consecutive_days(pools):
    wins, _, _, _ = build_controlled("no_drift", pools, STREAM, seed=0)
    ids = [w["window_id"] for w in wins]
    assert ids[0] == 10957 and ids == list(range(10957, 10957 + STREAM["n_windows"]))  # 2000-01-01


def test_pool_exhaustion_raises(pools):
    with pytest.raises(ValueError, match="exhausted"):
        build_controlled("volume", pools, {**STREAM, "window_size": 100}, seed=0)


def test_evaluate_delay_and_false_alarms():
    truth = {"concept_drift": True, "drift_start": {"position": 8}, "drift_end": {"position": 8}, "volume_change": None}
    alarm = [False] * 20
    alarm[6] = alarm[10] = True
    scores = [{"driftlens_alarm": a, "mcddd_alarm": False} for a in alarm]
    m = evaluate(truth, scores, n_ref=5, tolerance=3)
    assert m["driftlens"]["false_alarms_before_drift"] == 1
    assert m["driftlens"]["detection_delay"] == 2 and m["driftlens"]["detected_within_tolerance"]
    assert m["mcddd"]["first_alarm_after_drift"] is None and not m["mcddd"]["detected_within_tolerance"]
