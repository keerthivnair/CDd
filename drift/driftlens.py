"""DriftLens core for Task 3: PCA-reduced embedding distributions compared with the Frechet distance.

Pure numpy/scipy/sklearn; no model loading or file IO except save/load of the fitted reference.
"""
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA


def gaussian_stats(x: np.ndarray):
    """Mean vector and covariance matrix (rows = samples)."""
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0), np.cov(x, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """FDD = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2))."""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = np.real(covmean)  # imaginary parts are numerical noise for PSD inputs
    fdd = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(max(fdd, 0.0))


def subsample(x: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 0 or len(x) <= size:
        return x
    return x[rng.choice(len(x), size=size, replace=False)]


class DriftLens:
    """Fit on reference embeddings, calibrate a threshold, then score windows.

    The reference set is split into a *baseline* part (PCA + mean/covariance) and a held-out
    *threshold* part from which random in-distribution windows are drawn to calibrate the threshold.
    With `groups` (the reference window each tweet came from), every calibration window is drawn
    from a single group, so the threshold reflects normal window-to-window variation instead of
    the smaller distance of a random mix of all reference windows.
    """

    def __init__(self, pca_components: int = 150, threshold_fraction: float = 0.5,
                 n_calibration: int = 1000, percentile: float = 99.0, window_sample_size: int = 1000,
                 seed: int = 42):
        self.pca_components = pca_components
        self.threshold_fraction = threshold_fraction
        self.n_calibration = n_calibration
        self.percentile = percentile
        self.window_sample_size = window_sample_size
        self.seed = seed
        self.pca_mean = None            # PCA fitted on baseline, kept as plain arrays
        self.pca_basis = None
        self.explained_variance = None
        self.mu_b = None
        self.sigma_b = None
        self.threshold_pool = None      # PCA-projected held-out reference embeddings
        self.threshold_groups = None    # reference window of each threshold_pool row
        self.thresholds = {}            # window size -> threshold (FDD depends on sample size)
        self.calibration_scores = {}    # window size -> calibration FDD distribution

    # Steps 3-5: reference split, PCA, baseline mean/covariance
    def fit(self, reference_embeddings: np.ndarray, groups=None) -> "DriftLens":
        rng = np.random.default_rng(self.seed)
        x = np.asarray(reference_embeddings)
        groups = np.zeros(len(x), dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
        idx = rng.permutation(len(x))
        n_thr = int(round(len(x) * self.threshold_fraction))
        thr_idx, base_idx = idx[:n_thr], idx[n_thr:]
        if len(base_idx) < 2 or len(thr_idx) < 2:
            raise ValueError(f"Reference set too small ({len(x)} embeddings)")

        n_comp = min(self.pca_components, x.shape[1], len(base_idx) - 1)
        pca = PCA(n_components=n_comp, random_state=self.seed).fit(x[base_idx])
        self.pca_mean, self.pca_basis = pca.mean_, pca.components_
        self.explained_variance = float(pca.explained_variance_ratio_.sum())
        self.mu_b, self.sigma_b = gaussian_stats(self.project(x[base_idx]))
        self.threshold_pool = self.project(x[thr_idx])
        self.threshold_groups = groups[thr_idx]
        self.thresholds, self.calibration_scores = {}, {}
        return self

    def project(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.pca_mean) @ self.pca_basis.T

    def distance(self, projected: np.ndarray) -> float:
        mu_w, sigma_w = gaussian_stats(projected)
        return frechet_distance(self.mu_b, self.sigma_b, mu_w, sigma_w)

    # Step 8: threshold = percentile of FDDs of random in-distribution windows of the same size
    def calibrate(self, size: int) -> float:
        size = min(size, len(self.threshold_pool))
        if size not in self.thresholds:
            rng = np.random.default_rng([self.seed, size])
            ids, counts = np.unique(self.threshold_groups, return_counts=True)
            eligible = ids[counts >= size]
            pools = ([self.threshold_pool[self.threshold_groups == g] for g in eligible]
                     if len(eligible) else [self.threshold_pool])  # no group is big enough -> pooled
            scores = np.array([
                self.distance(subsample(pools[i % len(pools)], size, rng)) for i in range(self.n_calibration)
            ])
            self.calibration_scores[size] = scores
            self.thresholds[size] = float(np.percentile(scores, self.percentile))
        return self.thresholds[size]

    # Steps 6, 7, 9: window stats, FDD, alarm
    def score(self, embeddings: np.ndarray, window_id: int = 0) -> dict:
        rng = np.random.default_rng([self.seed, int(window_id)])
        sample = subsample(np.asarray(embeddings), self.window_sample_size, rng)
        n_used = min(len(sample), len(self.threshold_pool))
        fdd = self.distance(self.project(sample[:n_used]))
        threshold = self.calibrate(n_used)
        return {"driftlens_score": fdd, "driftlens_alarm": bool(fdd > threshold),
                "threshold": threshold, "n_used": int(n_used)}

    def save(self, path: str, **meta) -> None:
        np.savez_compressed(
            path,
            pca_basis=self.pca_basis, pca_mean=self.pca_mean,
            explained_variance=np.float64(self.explained_variance),
            mu_b=self.mu_b, sigma_b=self.sigma_b, threshold_pool=self.threshold_pool, threshold_groups=self.threshold_groups,
            threshold_sizes=np.array(list(self.thresholds), dtype=np.int64),
            threshold_values=np.array(list(self.thresholds.values()), dtype=np.float64),
            params=np.array([self.pca_components, self.threshold_fraction, self.n_calibration,
                             self.percentile, self.window_sample_size, self.seed], dtype=np.float64),
            **{f"meta_{k}": np.asarray(v) for k, v in meta.items()},
        )

    @classmethod
    def load(cls, path: str):
        d = np.load(path, allow_pickle=False)
        p = d["params"]
        dl = cls(int(p[0]), float(p[1]), int(p[2]), float(p[3]), int(p[4]), int(p[5]))
        dl.pca_basis, dl.pca_mean = d["pca_basis"], d["pca_mean"]
        dl.explained_variance = float(d["explained_variance"])
        dl.mu_b, dl.sigma_b = d["mu_b"], d["sigma_b"]
        dl.threshold_pool = d["threshold_pool"]
        dl.threshold_groups = d["threshold_groups"]
        dl.thresholds = dict(zip(d["threshold_sizes"].tolist(), d["threshold_values"].tolist()))
        meta = {k[5:]: d[k] for k in d.files if k.startswith("meta_")}
        return dl, meta
