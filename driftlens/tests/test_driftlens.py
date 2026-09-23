import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from driftlens import DriftLens, frechet_distance, gaussian_stats  # noqa: E402
from embedder import prepare_texts  # noqa: E402
from run_driftlens import load_windows  # noqa: E402

DIM = 32


def gaussian(n, shift=0.0, seed=0):
    return np.random.default_rng(seed).normal(size=(n, DIM)) + shift


@pytest.fixture(scope="module")
def fitted():
    return DriftLens(pca_components=16, n_calibration=200, window_sample_size=300, seed=1).fit(gaussian(6000))


def test_fdd_identical_distributions_is_zero():
    mu, sigma = gaussian_stats(gaussian(2000))
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)


def test_fdd_symmetric_and_grows_with_shift():
    a = gaussian_stats(gaussian(3000, seed=1))
    b = gaussian_stats(gaussian(3000, shift=0.5, seed=2))
    c = gaussian_stats(gaussian(3000, shift=1.0, seed=3))
    assert frechet_distance(*a, *b) == pytest.approx(frechet_distance(*b, *a), rel=1e-6)
    assert frechet_distance(*a, *c) > frechet_distance(*a, *b) > 0


def test_fdd_mean_term_matches_closed_form():
    # Same covariance, mean shifted by s in every dimension -> FDD ~= DIM * s^2
    mu = np.zeros(DIM)
    sigma = np.eye(DIM)
    assert frechet_distance(mu, sigma, mu + 0.5, sigma) == pytest.approx(DIM * 0.25, rel=1e-6)


def test_pca_components_clipped_to_data():
    dl = DriftLens(pca_components=500, n_calibration=10, window_sample_size=20).fit(gaussian(60))
    assert dl.pca_basis.shape[0] <= DIM


def test_threshold_covers_in_distribution_windows(fitted):
    thr = fitted.calibrate(300)
    assert np.mean(fitted.calibration_scores[300] <= thr) >= 0.98


def test_alarm_fires_on_shift_only(fitted):
    normal = fitted.score(gaussian(1000, seed=7), window_id=1)
    drifted = fitted.score(gaussian(1000, shift=0.5, seed=8), window_id=2)
    assert not normal["driftlens_alarm"]
    assert drifted["driftlens_alarm"]
    assert drifted["driftlens_score"] > normal["driftlens_score"]
    assert normal["n_used"] == 300


def test_grouped_calibration_accepts_normal_window_variation():
    # Each reference "day" has its own small topic offset; a new day with a similar offset is normal.
    rng = np.random.default_rng(3)
    offsets = rng.normal(scale=0.15, size=(8, DIM))
    ref = np.vstack([gaussian(1500, seed=20 + i) + offsets[i] for i in range(8)])
    groups = np.repeat(np.arange(8), 1500)
    new_day = gaussian(1000, seed=99) + rng.normal(scale=0.15, size=DIM)

    kw = dict(pca_components=16, n_calibration=200, window_sample_size=500, seed=1)
    pooled = DriftLens(**kw).fit(ref)
    grouped = DriftLens(**kw).fit(ref, groups)
    assert pooled.score(new_day, window_id=1)["driftlens_alarm"]
    assert not grouped.score(new_day, window_id=1)["driftlens_alarm"]
    assert grouped.score(new_day + 0.5, window_id=1)["driftlens_alarm"]


def test_small_window_gets_its_own_threshold(fitted):
    r = fitted.score(gaussian(100, seed=9), window_id=3)
    assert r["n_used"] == 100
    assert r["threshold"] > fitted.calibrate(300)  # fewer samples -> noisier covariance -> higher FDD
    assert not r["driftlens_alarm"]


def test_score_is_deterministic(fitted):
    x = gaussian(1000, seed=10)
    assert fitted.score(x, window_id=5) == fitted.score(x, window_id=5)


def test_save_load_roundtrip(fitted, tmp_path):
    x = gaussian(1000, shift=0.2, seed=11)
    before = fitted.score(x, window_id=6)
    path = str(tmp_path / "ref.npz")
    fitted.save(path, model="m", reference_ids=np.array([1, 2]))
    dl, meta = DriftLens.load(path)
    assert str(meta["model"]) == "m" and meta["reference_ids"].tolist() == [1, 2]
    assert dl.score(x, window_id=6) == before


def test_prepare_texts_strips_replacement_char():
    assert prepare_texts(["it�s fine", "�", " ok "]) == ["its fine", "ok"]


def test_load_windows_file_and_folder(tmp_path):
    rows = [{"window_id": 3, "texts": ["c"]}, {"window_id": 1, "texts": ["a"]}]
    f = tmp_path / "all.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert [w["window_id"] for w in load_windows(str(f))] == [1, 3]

    for i, r in enumerate(rows):
        d = tmp_path / "windows" / f"batch_id={i}"
        d.mkdir(parents=True)
        (d / "part-0000.json").write_text(json.dumps(r) + "\n", encoding="utf-8")
    assert [w["window_id"] for w in load_windows(str(tmp_path / "windows"))] == [1, 3]
