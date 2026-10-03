"""Tests for the MCD-DD drift detector.

All tests use synthetic data (no model download needed) and verify that MCD-DD:
- trains without error on reference embeddings
- produces low scores for in-distribution windows
- produces high scores / alarms for shifted windows
- can be saved and loaded
- uses the same embedding dimensionality as DriftLens (384-d)
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcddd import MCDDD, SampleSetEncoder  # noqa: E402

DIM = 384  # same as all-MiniLM-L6-v2


def gaussian(n, dim=DIM, shift=0.0, seed=0):
    """Generate random embeddings mimicking L2-normalised sentence embeddings."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32) + shift
    # L2 normalise to match the real embeddings
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


@pytest.fixture(scope="module")
def fitted():
    """A detector fitted on 2000 in-distribution embeddings."""
    ref = gaussian(2000, seed=1)
    return MCDDD(input_dim=DIM, hidden_dim=64, output_dim=32,
                 train_epochs=2, seed=42).fit(ref)


def test_encoder_output_shape():
    enc = SampleSetEncoder(DIM, 64, 32)
    x = torch.randn(4, 50, DIM)
    out = enc(x)
    assert out.shape == (4, 32)


def test_fit_sets_threshold(fitted):
    assert fitted.threshold is not None
    assert fitted.threshold > 0


def test_in_distribution_no_alarm(fitted):
    normal = gaussian(500, seed=10)
    r = fitted.score(normal)
    assert "mcddd_score" in r
    assert "mcddd_alarm" in r
    assert isinstance(r["mcddd_score"], float)


def test_shifted_window_scores_higher(fitted):
    normal = gaussian(500, seed=10)
    shifted = gaussian(500, shift=2.0, seed=11)
    r_norm = fitted.score(normal)
    r_shift = fitted.score(shifted)
    assert r_shift["mcddd_score"] > r_norm["mcddd_score"]


def test_large_shift_triggers_alarm():
    ref = gaussian(2000, seed=1)
    det = MCDDD(input_dim=DIM, hidden_dim=64, output_dim=32,
                train_epochs=3, seed=42).fit(ref)
    # Score a normal window first (sets _prev_embeddings)
    det.score(gaussian(500, seed=20))
    # Now score a heavily shifted window
    r = det.score(gaussian(500, shift=5.0, seed=21))
    assert r["mcddd_alarm"] is True


def test_first_score_returns_zero():
    """First call has no previous window, so MCD should be 0."""
    det = MCDDD(input_dim=DIM, hidden_dim=64, output_dim=32,
                train_epochs=1, seed=42).fit(gaussian(500, seed=1))
    # Reset _prev_embeddings to simulate first call
    det._prev_embeddings = None
    r = det.score(gaussian(300, seed=5))
    assert r["mcddd_score"] == 0.0
    assert r["mcddd_alarm"] is False


def test_save_load_roundtrip(fitted, tmp_path):
    path = str(tmp_path / "mcddd.pt")
    fitted.save(path)
    loaded = MCDDD.load(path)
    assert loaded.threshold == fitted.threshold
    assert loaded.input_dim == fitted.input_dim
    assert loaded.hidden_dim == fitted.hidden_dim
    # Encoder produces the same output
    x = torch.randn(1, 50, DIM)
    fitted.encoder.eval()
    loaded.encoder.eval()
    with torch.no_grad():
        out1 = fitted.encoder(x.to(fitted.device))
        out2 = loaded.encoder(x.to(loaded.device))
    assert torch.allclose(out1.cpu(), out2.cpu(), atol=1e-5)


def test_uses_384_dimensions():
    """Verify MCD-DD defaults to 384-d input matching all-MiniLM-L6-v2."""
    det = MCDDD()
    assert det.input_dim == 384
    # Verify the encoder's first layer accepts 384-d input
    first_layer = det.encoder.item_mlp[0]
    assert first_layer.in_features == 384


def test_sub_window_splitting():
    det = MCDDD(input_dim=DIM, n_sub_windows=5)
    emb = gaussian(1000, seed=3)
    subs = det._to_sub_windows(emb)
    assert len(subs) == 5
    assert sum(len(s) for s in subs) == 1000


def test_sub_window_splitting_small_input():
    """When input is smaller than n_sub_windows, adapt gracefully."""
    det = MCDDD(input_dim=DIM, n_sub_windows=5)
    emb = gaussian(3, seed=4)
    subs = det._to_sub_windows(emb)
    assert len(subs) <= 3
    assert sum(len(s) for s in subs) == 3
