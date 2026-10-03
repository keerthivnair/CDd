"""MCD-DD: Online Drift Detection with Maximum Concept Discrepancy.

Adapted to consume the same pre-computed 384-d L2-normalised embeddings
that DriftLens uses (sentence-transformers/all-MiniLM-L6-v2, cached by embedder.py).

No model loading or separate embedding computation — this module works with
plain numpy arrays of shape (n_samples, 384).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger("mcddd")


class SampleSetEncoder(nn.Module):
    """Deep Sets: permutation-invariant encoding of a sample set to a single vector."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.item_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.set_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, set_size, input_dim)
        item_features = self.item_mlp(x)
        pooled = torch.mean(item_features, dim=1)
        return self.set_mlp(pooled)


class MCDDD:
    """MCD-DD detector that operates on pre-computed embeddings.

    Parameters
    ----------
    input_dim : int
        Embedding dimensionality (384 for all-MiniLM-L6-v2).
    hidden_dim : int
        Hidden layer width in the Deep Sets encoder.
    output_dim : int
        Concept embedding dimensionality produced by the encoder.
    lr : float
        Adam learning rate for online contrastive updates.
    lambda_gp : float
        Weight of the gradient penalty (Lipschitz regulariser).
    lipschitz_k : float
        Target gradient norm for the penalty.
    eps_small : float
        Noise scale for weak-negative pairs.
    eps_big : float
        Noise scale for strong-negative pairs.
    n_sub_windows : int
        Number of sub-windows to slice a reference/training window into.
    sample_set_size : int
        Number of items sampled per set when generating contrastive pairs.
    k_pairs : int
        Number of contrastive pairs drawn per sub-window.
    train_epochs : int
        Gradient steps per ``online_update`` call.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        output_dim: int = 64,
        lr: float = 1e-3,
        lambda_gp: float = 0.1,
        lipschitz_k: float = 1.0,
        eps_small: float = 0.01,
        eps_big: float = 0.1,
        n_sub_windows: int = 5,
        sample_set_size: int = 100,
        k_pairs: int = 5,
        train_epochs: int = 1,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.lambda_gp = lambda_gp
        self.lipschitz_k = lipschitz_k
        self.eps_small = eps_small
        self.eps_big = eps_big
        self.n_sub_windows = n_sub_windows
        self.sample_set_size = sample_set_size
        self.k_pairs = k_pairs
        self.train_epochs = train_epochs
        self.seed = seed

        device = "cpu"
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                device = "cuda"
            except Exception:
                device = "cpu"
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.encoder = SampleSetEncoder(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.threshold: float | None = None
        self._prev_embeddings: np.ndarray | None = None  # previous window for drift comparison

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_sub_windows(self, embeddings: np.ndarray) -> list[np.ndarray]:
        """Split a flat embedding array into roughly equal sub-windows."""
        n = len(embeddings)
        k = min(self.n_sub_windows, n)
        if k < 2:
            return [embeddings]
        splits = np.array_split(embeddings, k)
        return [s for s in splits if len(s) > 0]

    def _gradient_penalty(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        out = self.encoder(x)
        grads = torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.reshape(grads.size(0), -1)
        return ((grads.norm(2, dim=1) - self.lipschitz_k) ** 2).mean()

    def _generate_pairs(self, sub_windows: list[np.ndarray]):
        """Contrastive positive / weak-neg / strong-neg pairs from sub-windows."""
        rng = np.random.default_rng(self.seed)
        pos_1, pos_2 = [], []
        wn_1, wn_2 = [], []
        sn_1, sn_2 = [], []

        n_sub = len(sub_windows)
        m = self.sample_set_size

        for j in range(n_sub):
            sw = sub_windows[j]
            n = len(sw)
            if n == 0:
                continue

            for _ in range(self.k_pairs):
                idx1 = rng.choice(n, m, replace=True)
                idx2 = rng.choice(n, m, replace=True)
                s1, s2 = sw[idx1], sw[idx2]

                # Positive pair: two samples from the same sub-window
                pos_1.append(s1)
                pos_2.append(s2)

                # Weak negative: small noise
                noise = rng.normal(0, self.eps_small, s2.shape).astype(np.float32)
                wn_1.append(s1)
                wn_2.append(s2 + noise)

                # Strong negative: distant sub-window + large noise
                distant_j = 0 if j == n_sub - 1 else n_sub - 1
                dsw = sub_windows[distant_j]
                idx_d = rng.choice(len(dsw), m, replace=True)
                s_d = dsw[idx_d]
                noise_big = rng.normal(0, self.eps_big, s_d.shape).astype(np.float32)
                sn_1.append(s1)
                sn_2.append(s_d + noise_big)

        def _t(arrs):
            return torch.tensor(np.array(arrs), dtype=torch.float32).to(self.device)

        return _t(pos_1), _t(pos_2), _t(wn_1), _t(wn_2), _t(sn_1), _t(sn_2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference_embeddings: np.ndarray) -> "MCDDD":
        """Offline training on reference embeddings (same ones used by DriftLens).

        Splits the reference into sub-windows, generates contrastive pairs,
        and trains the encoder for ``train_epochs`` steps.  Also sets the
        initial threshold and stores the reference as the first "previous"
        window for drift comparison.
        """
        ref = np.asarray(reference_embeddings, dtype=np.float32)
        sub_windows = self._to_sub_windows(ref)

        for _ in range(self.train_epochs):
            self._train_step(sub_windows)

        self._prev_embeddings = ref
        log.info(
            "MCD-DD fit complete: %s reference embeddings, %s sub-windows, threshold=%.6f",
            len(ref), len(sub_windows), self.threshold,
        )
        return self

    def _train_step(self, sub_windows: list[np.ndarray]) -> None:
        """Single contrastive training step with gradient penalty."""
        self.encoder.train()
        p1, p2, wn1, wn2, sn1, sn2 = self._generate_pairs(sub_windows)

        h_p1, h_p2 = self.encoder(p1), self.encoder(p2)
        h_wn1, h_wn2 = self.encoder(wn1), self.encoder(wn2)
        h_sn1, h_sn2 = self.encoder(sn1), self.encoder(sn2)

        mcd_p = torch.sum((h_p1 - h_p2) ** 2, dim=1)
        mcd_wn = torch.sum((h_wn1 - h_wn2) ** 2, dim=1)
        mcd_sn = torch.sum((h_sn1 - h_sn2) ** 2, dim=1)

        exp_p = torch.exp(-mcd_p)
        exp_wn = torch.exp(-mcd_wn)
        exp_sn = torch.exp(-mcd_sn)

        loss_contrastive = -torch.log(exp_p / (exp_p + exp_wn + exp_sn + 1e-12)).mean()
        loss_gp = self._gradient_penalty(p1)
        total_loss = loss_contrastive + self.lambda_gp * loss_gp

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Dynamic threshold: 95th percentile of positive-pair MCDs
        self.threshold = torch.quantile(mcd_p.detach(), 0.95).item()

    def score(self, window_embeddings: np.ndarray) -> dict:
        """Score a new window against the previous one.

        Uses the same pre-computed embeddings that DriftLens scores.
        Returns ``mcddd_score`` and ``mcddd_alarm``.
        """
        curr = np.asarray(window_embeddings, dtype=np.float32)

        if self._prev_embeddings is None:
            # No previous window yet (first call); cannot compute drift.
            self._prev_embeddings = curr
            return {"mcddd_score": 0.0, "mcddd_alarm": False, "mcddd_threshold": 0.0}

        # Online update: train on the sub-windows of the current window
        sub_windows = self._to_sub_windows(curr)
        if len(sub_windows) >= 2:
            self._train_step(sub_windows)

        # Detect drift: compare previous vs current window
        mcd, is_drift = self._detect(self._prev_embeddings, curr)

        # Slide: current becomes previous for the next call
        self._prev_embeddings = curr

        return {
            "mcddd_score": float(mcd),
            "mcddd_alarm": bool(is_drift),
            "mcddd_threshold": float(self.threshold) if self.threshold else 0.0,
        }

    def _detect(self, prev: np.ndarray, curr: np.ndarray) -> tuple[float, bool]:
        """Compute MCD between two sets of embeddings."""
        self.encoder.eval()
        with torch.no_grad():
            t_prev = torch.tensor(prev, dtype=torch.float32).unsqueeze(0).to(self.device)
            t_curr = torch.tensor(curr, dtype=torch.float32).unsqueeze(0).to(self.device)
            h_prev = self.encoder(t_prev)
            h_curr = self.encoder(t_curr)
            mcd = torch.sum((h_prev - h_curr) ** 2).item()
            is_drift = mcd > self.threshold if self.threshold else False
        return mcd, is_drift

    def save(self, path: str) -> None:
        """Persist encoder weights and threshold."""
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "threshold": self.threshold,
                "params": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "lr": self.lr,
                    "lambda_gp": self.lambda_gp,
                    "lipschitz_k": self.lipschitz_k,
                    "eps_small": self.eps_small,
                    "eps_big": self.eps_big,
                    "n_sub_windows": self.n_sub_windows,
                    "sample_set_size": self.sample_set_size,
                    "k_pairs": self.k_pairs,
                    "train_epochs": self.train_epochs,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MCDDD":
        data = torch.load(path, map_location="cpu", weights_only=False)
        p = data["params"]
        obj = cls(**p)
        obj.encoder.load_state_dict(data["encoder_state"])
        obj.optimizer.load_state_dict(data["optimizer_state"])
        obj.threshold = data["threshold"]
        obj.encoder.to(obj.device)
        return obj
