"""MCD-DD: Online Drift Detection with Maximum Concept Discrepancy (Wan et al., KDD 2024).

Adapted to consume the same pre-computed 384-d L2-normalised embeddings that DriftLens uses
(sentence-transformers/all-MiniLM-L6-v2, cached by embedder.py). Works on plain numpy arrays
of shape (n_samples, 384); runs on the GPU when one is usable, else the CPU.

How it maps onto the paper (sub-window = one stream window, i.e. one day):
- Sample-set encoder: Deep Sets (item MLP -> mean pool -> set MLP), trained with the paper's
  InfoNCE loss on positive pairs (two sample sets from the same day), weak negatives (same day +
  small Gaussian noise) and strong negatives (a temporally distant day + large noise).
- MCD between two windows = ||h(S_a) - h(S_b)||_2 of their concept embeddings (paper Eq. 6-7).
- Detection: the newest window is compared with every window in its sliding context (the previous
  `context_windows` windows, i.e. the paper's sliding window W) and the score is the maximum of those
  MCDs. Comparing against the whole context, as in the paper's Fig. 3 heatmaps, is what lets
  gradual drift accumulate into a detectable gap; comparing only adjacent windows cannot see it.
- Threshold: bootstrapped from in-distribution reference data at the exact set size and context
  length used for scoring (paper Sec. 4.3), then frozen. See `calibrate()`.

Deviations from the paper (deliberate, documented in drift/README.md):
- Inputs are standardised with the reference mean/std so the noise scales and the encoder see
  O(1) features (raw MiniLM dims have std ~0.05, which made MCD values ~1e-6).
- The encoder is trained offline on the reference period; online updates are optional
  (`online_training`) because updating every window inflated the threshold in earlier runs.
"""

import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger("mcddd")


def pick_device(device: str = "") -> torch.device:
    """'' = auto: CUDA if a kernel actually runs on it (an unsupported GPU arch only fails at launch)."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return torch.device("cuda")
        except Exception as ex:  # e.g. sm_120 GPU with a torch build that lacks sm_120 kernels
            log.warning("CUDA present but unusable by this torch build (%s); MCD-DD falls back to CPU", ex)
    return torch.device("cpu")


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


class EnsembleEncoder(nn.Module):
    """n independently initialised SampleSetEncoders; the concept vector is their concatenation / sqrt(n).

    MCD in the joint space is then the RMS of the members' MCDs. Each member is trained on its own
    InfoNCE loss (see MCDDD._train_step), so averaging removes the luck of a single random start.
    """

    def __init__(self, n: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.members = nn.ModuleList(SampleSetEncoder(input_dim, hidden_dim, output_dim) for _ in range(n))

    def per_member(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members], dim=1)  # (batch, n, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.per_member(x)
        return h.flatten(1) / len(self.members) ** 0.5


class MCDDD:
    """MCD-DD detector that operates on pre-computed embeddings.

    Parameters
    ----------
    input_dim, hidden_dim, output_dim : int
        Encoder sizes (input 384 for all-MiniLM-L6-v2).
    n_encoders : int
        Ensemble size. Independently initialised encoders, each trained on its own loss; scores use
        their joint concept vector. One encoder's result depended heavily on its random start.
    lr : float
        Adam learning rate.
    lambda_gp, lipschitz_k : float
        Gradient-penalty weight and target per-item gradient norm (paper Eq. 14). 0 disables it.
    eps_small, eps_big : float
        Std of the Gaussian noise for weak / strong negatives, in standardised units.
    n_sub_windows : int
        Sub-windows to cut the data into when no per-window grouping is given.
    sample_set_size : int
        Items per sample set in contrastive training pairs (paper m).
    k_pairs : int
        Contrastive pairs per sub-window per training step (paper k).
    train_epochs : int
        Gradient steps during offline fitting; each step draws fresh pairs.
    scoring_sample_size : int
        Max items per window used for scoring (windows are randomly subsampled to this).
    context_windows : int
        How many previous windows the newest one is compared with (paper sliding window W - 1).
    n_calibration, percentile : int, float
        Bootstrap draws and percentile of the in-distribution MCD used as threshold.
    online_training : bool
        Take one training step per scored window on its sliding context, then recalibrate.
    holdout_fraction : float
        Share of every reference window kept out of training, used for calibration and the initial
        context (needs to hold (context_windows + 1) * scoring_sample_size rows for disjoint draws).
    seed : int
        Random seed for reproducibility.
    device : str
        "" = auto (CUDA if usable), or "cpu" / "cuda".
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        output_dim: int = 64,
        n_encoders: int = 5,
        lr: float = 1e-3,
        lambda_gp: float = 1.0,
        lipschitz_k: float = 1.0,
        eps_small: float = 0.03,
        eps_big: float = 0.3,
        n_sub_windows: int = 5,
        sample_set_size: int = 100,
        k_pairs: int = 10,
        train_epochs: int = 100,
        scoring_sample_size: int = 1000,
        context_windows: int = 10,
        n_calibration: int = 500,
        percentile: float = 99.0,
        online_training: bool = False,
        holdout_fraction: float = 0.8,
        seed: int = 42,
        device: str = "",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_encoders = n_encoders
        self.lr = lr
        self.lambda_gp = lambda_gp
        self.lipschitz_k = lipschitz_k
        self.eps_small = eps_small
        self.eps_big = eps_big
        self.n_sub_windows = n_sub_windows
        self.sample_set_size = sample_set_size
        self.k_pairs = k_pairs
        self.train_epochs = train_epochs
        self.scoring_sample_size = scoring_sample_size
        self.context_windows = context_windows
        self.n_calibration = n_calibration
        self.percentile = percentile
        self.online_training = online_training
        self.holdout_fraction = holdout_fraction
        self.seed = seed

        self.device = pick_device(device)
        torch.manual_seed(seed)
        self._gen = torch.Generator(device=self.device).manual_seed(seed)  # training pairs: fresh every step
        self.encoder = EnsembleEncoder(n_encoders, input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=lr)

        self.feat_mean: np.ndarray | None = None   # reference standardisation
        self.feat_std: np.ndarray | None = None
        self.ref_pool: np.ndarray | None = None    # reference embeddings (float16) for calibration
        self.ref_groups: np.ndarray | None = None  # reference window of each ref_pool row, in time order
        self.thresholds: dict = {}                 # (set size, context length) -> threshold
        self.threshold: float | None = None        # threshold at full scoring size + full context
        self.last_loss: float | None = None
        self._context: deque = deque(maxlen=context_windows)  # previous windows' (subsampled) embeddings
        self._init_context: list = []  # context right after fit (held-out reference rows)
        self._n_scored = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _std(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x if self.feat_mean is None else (x - self.feat_mean) / self.feat_std

    def _tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def _encode(self, sets: np.ndarray, chunk: int = 64) -> torch.Tensor:
        """(B, n, d) standardised sets (array or device tensor) -> (B, output_dim), in chunks."""
        self.encoder.eval()
        sets = sets if torch.is_tensor(sets) else self._tensor(sets)
        return torch.cat([self.encoder(sets[i:i + chunk]) for i in range(0, len(sets), chunk)])

    def _to_sub_windows(self, embeddings: np.ndarray) -> list[np.ndarray]:
        """Split a flat embedding array into roughly equal sub-windows."""
        n = len(embeddings)
        k = min(self.n_sub_windows, n)
        if k < 2:
            return [embeddings]
        splits = np.array_split(embeddings, k)
        return [s for s in splits if len(s) > 0]

    def _gradient_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Per-item gradient-norm penalty (paper Eq. 14), on a random projection of the output.

        The set is a mean of m items, so d f / d x_i carries a 1/m factor; multiplying by m gives
        the item-level gradient that the Lipschitz constant L refers to.
        """
        x = x.detach().requires_grad_(True)
        out = self.encoder(x)
        v = torch.randn_like(out)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grads = torch.autograd.grad((out * v).sum(), x, create_graph=True)[0] * x.shape[1]
        return ((grads.norm(2, dim=2) - self.lipschitz_k) ** 2).mean()

    def _generate_pairs(self, sub_windows: list[np.ndarray]):
        """Contrastive anchor / positive / weak-negative / strong-negative sets (paper Sec. 4.4).

        sub_windows are standardised and in time order. Sampling and noise run on the device, with
        a generator that keeps advancing, so every training step sees fresh pairs.
        """
        n_sub, m, k = len(sub_windows), self.sample_set_size, self.k_pairs
        g = self._gen
        sw = [s if torch.is_tensor(s) else self._tensor(s) for s in sub_windows]

        def draw(x):  # k sets of m items (with replacement, as in a bootstrap)
            return x[torch.randint(len(x), (k, m), generator=g, device=self.device)]

        a, pos, wn, sn = [], [], [], []
        for j in range(n_sub):
            # Strong negatives come from a temporally distant sub-window (half the context away).
            far = sw[(j + max(1, n_sub // 2)) % n_sub]
            a.append(draw(sw[j]))
            pos.append(draw(sw[j]))
            wn.append(draw(sw[j]))
            sn.append(draw(far))
        a, pos, wn, sn = (torch.cat(t) for t in (a, pos, wn, sn))
        wn = wn + self.eps_small * torch.randn(wn.shape, generator=g, device=self.device)
        sn = sn + self.eps_big * torch.randn(sn.shape, generator=g, device=self.device)
        return a, pos, wn, sn

    def _train_step(self, sub_windows: list[np.ndarray]) -> float:
        """One InfoNCE step on fresh pairs: pull positives together, push both negatives apart."""
        self.encoder.train()
        a, pos, wn, sn = self._generate_pairs(sub_windows)
        h_a = self.encoder.per_member(a)  # (pairs, n_encoders, output_dim): every member has its own loss
        mcd_p = (h_a - self.encoder.per_member(pos)).norm(dim=2)
        mcd_wn = (h_a - self.encoder.per_member(wn)).norm(dim=2)
        mcd_sn = (h_a - self.encoder.per_member(sn)).norm(dim=2)
        # -log( exp(-MCD_p) / (exp(-MCD_p) + exp(-MCD_wn) + exp(-MCD_sn)) ), computed stably
        logits = -torch.stack([mcd_p, mcd_wn, mcd_sn], dim=2)
        loss = (torch.logsumexp(logits, dim=2) - logits[..., 0]).mean(dim=0).sum()
        if self.lambda_gp > 0:
            loss = loss + self.lambda_gp * self._gradient_penalty(a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.item()) / self.n_encoders  # per-encoder average, for logging
        return self.last_loss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference_embeddings: np.ndarray, groups=None) -> "MCDDD":
        """Offline training on the reference period, then threshold calibration.

        groups: reference window id of every row (time-ordered ids). With it each reference window is
        one sub-window, as in the paper; without it the reference is cut into `n_sub_windows` slices.

        Every reference window is split: `1 - holdout_fraction` of its rows train the encoder, the rest
        are held out. Calibration and the initial sliding context use only held-out rows, because the
        encoder memorises its training tweets and makes them look far from everything else (scoring
        windows next to the training data then alarmed on every window).
        """
        ref = np.asarray(reference_embeddings, dtype=np.float32)
        groups = np.zeros(len(ref), dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
        self.feat_mean = ref.mean(axis=0)
        self.feat_std = ref.std(axis=0) + 1e-6
        ref_s = self._std(ref)

        rng = np.random.default_rng(self.seed)
        ids = np.unique(groups)
        train_rows, held_rows = [], []
        for g in ids:
            rows = rng.permutation(np.flatnonzero(groups == g))
            n_held = int(round(len(rows) * self.holdout_fraction))
            held_rows.append(rows[:n_held])
            train_rows.append(rows[n_held:])
        held = np.concatenate(held_rows)
        if len(held) < 2 or sum(len(r) for r in train_rows) < 2:
            raise ValueError(f"Reference set too small ({len(ref)} embeddings)")

        if len(ids) >= 2:
            sub_windows = [ref_s[r] for r in train_rows if len(r)]
        else:
            sub_windows = self._to_sub_windows(ref_s[np.sort(train_rows[0])])
        sub_windows = [self._tensor(sw) for sw in sub_windows]  # copy to the device once, not every step
        for _ in range(self.train_epochs):
            self._train_step(sub_windows)

        self.ref_pool = ref[held].astype(np.float16)
        self.ref_groups = groups[held]
        self.thresholds = {}
        self.threshold = self.calibrate(min(self.scoring_sample_size, len(held)), self.context_windows)
        # The stream continues straight after the reference: seed the context with the held-out rows of
        # the last reference windows (`score()` should be called only for windows after the reference).
        self._context.clear()
        for r in held_rows[-self.context_windows:]:
            if len(r):
                self._context.append(ref_s[rng.permutation(r)[: self.scoring_sample_size]])
        self._init_context = [c.copy() for c in self._context]  # saved with the model (float32: reload is exact)
        self._n_scored = 0
        log.info("MCD-DD fit on %s: %s reference embeddings (%s held out), %s sub-windows, final loss %.4f, "
                 "threshold=%.4f", self.device, len(ref), len(held), len(sub_windows),
                 self.last_loss or float("nan"), self.threshold)
        return self

    def calibrate(self, size: int, n_context: int) -> float:
        """Threshold for comparing a `size`-item window with `n_context` previous windows.

        Bootstrap: draw n_context + 1 in-distribution sets of `size` items, take the max MCD between the
        last one and the others (the same statistic `score()` computes), repeat `n_calibration` times
        and keep the `percentile`. Sets come from distinct reference windows (so normal day-to-day
        variation is in the null) when enough windows hold >= 2*size items to subsample from; otherwise
        disjoint draws from the pooled reference (e.g. the experiment scenarios, whose reference
        windows are i.i.d. samples of one pool with exactly 1000 tweets each).
        """
        key = (int(size), int(n_context))
        if key in self.thresholds:
            return self.thresholds[key]
        rng = np.random.default_rng([self.seed, size, n_context])
        pool = self._tensor(self._std(self.ref_pool))  # on the device; sets are gathered there
        n_sets = n_context + 1
        ids, counts = np.unique(self.ref_groups, return_counts=True)
        eligible = ids[counts >= 2 * size]
        by_group = {g: np.flatnonzero(self.ref_groups == g) for g in eligible}

        scores = []
        # Bootstrap draws per batch, capped at ~128 MB of float32 sets so RAM/VRAM stay small.
        batch = max(1, (128 << 20) // (4 * n_sets * size * pool.shape[1]))
        for start in range(0, self.n_calibration, batch):
            sets = []
            for _ in range(min(batch, self.n_calibration - start)):
                if len(eligible) >= n_sets:
                    chosen = np.sort(rng.choice(eligible, n_sets, replace=False))
                    rows = [rng.choice(by_group[g], size, replace=False) for g in chosen]
                elif len(pool) >= n_sets * size:  # disjoint sets from the pooled reference
                    rows = np.split(rng.choice(len(pool), n_sets * size, replace=False), n_sets)
                else:  # pool too small for disjoint sets: independent draws (sets may share a few rows)
                    rows = [rng.choice(len(pool), size, replace=len(pool) < size) for _ in range(n_sets)]
                sets.extend(rows)
            idx = torch.as_tensor(np.stack(sets), device=self.device)
            h = self._encode(pool[idx]).view(-1, n_sets, self.n_encoders * self.output_dim)
            scores.append((h[:, :-1] - h[:, -1:]).norm(dim=2).max(dim=1).values.cpu().numpy())
        self.thresholds[key] = float(np.percentile(np.concatenate(scores), self.percentile))
        return self.thresholds[key]

    def score(self, window_embeddings: np.ndarray, window_id: int | None = None) -> dict:
        """Score a new window against its sliding context, then add it to the context.

        Call it for the windows after the reference period, in stream order (fit() already put the
        last reference windows in the context). Uses the same pre-computed embeddings as DriftLens. The window is randomly
        subsampled (seeded by window_id) to at most `scoring_sample_size` items, so traffic volume
        does not change the score's noise level. With an empty context: score 0, no alarm.
        """
        wid = self._n_scored if window_id is None else int(window_id)
        self._n_scored += 1
        curr = self._std(window_embeddings)
        rng = np.random.default_rng([self.seed, wid])
        if len(curr) > self.scoring_sample_size:
            curr = curr[rng.choice(len(curr), self.scoring_sample_size, replace=False)]
        else:
            curr = curr[rng.permutation(len(curr))]

        if not self._context or len(curr) == 0:
            if len(curr):
                self._context.append(curr)
            return {"mcddd_score": 0.0, "mcddd_alarm": False,
                    "mcddd_threshold": float(self.threshold or 0.0), "mcddd_lag": None}

        # All sets in one comparison share the smallest size; context windows are already shuffled.
        size = min(len(curr), min(len(c) for c in self._context))
        context = list(self._context)
        h = self._encode(np.stack([c[:size] for c in context] + [curr[:size]]))
        dists = (h[:-1] - h[-1:]).norm(dim=1).cpu().numpy()
        best = int(np.argmax(dists))
        mcd = float(dists[best])
        threshold = self.calibrate(size, len(context))

        if self.online_training:
            self._train_step(context + [curr])
            self.thresholds = {}  # encoder changed -> recalibrate lazily
            self.threshold = self.calibrate(min(self.scoring_sample_size, len(self.ref_pool)),
                                            self.context_windows)

        self._context.append(curr)
        return {
            "mcddd_score": mcd,
            "mcddd_alarm": bool(mcd > threshold),
            "mcddd_threshold": threshold,
            "mcddd_lag": len(context) - best,  # how many windows back the largest discrepancy is
        }

    def params(self) -> dict:
        return {k: getattr(self, k) for k in (
            "input_dim", "hidden_dim", "output_dim", "n_encoders", "lr", "lambda_gp", "lipschitz_k", "eps_small",
            "eps_big", "n_sub_windows", "sample_set_size", "k_pairs", "train_epochs",
            "scoring_sample_size", "context_windows", "n_calibration", "percentile",
            "online_training", "holdout_fraction", "seed")}

    def save(self, path: str, **meta) -> None:
        """Persist encoder, standardisation, reference pool, thresholds and any metadata."""
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "threshold": self.threshold,
                "thresholds": self.thresholds,
                "feat_mean": self.feat_mean,
                "feat_std": self.feat_std,
                "ref_pool": self.ref_pool,
                "ref_groups": self.ref_groups,
                "init_context": self._init_context,
                "params": self.params(),
                "meta": meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "") -> "MCDDD":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "feat_mean" not in data:
            raise ValueError("model file predates the current MCD-DD format; refit it")
        obj = cls(**data["params"], device=device)
        obj.encoder.load_state_dict(data["encoder_state"])
        obj.optimizer.load_state_dict(data["optimizer_state"])
        obj.threshold = data["threshold"]
        obj.thresholds = data["thresholds"]
        obj.feat_mean, obj.feat_std = data["feat_mean"], data["feat_std"]
        obj.ref_pool, obj.ref_groups = data["ref_pool"], data["ref_groups"]
        obj.meta = data.get("meta", {})
        obj._init_context = data["init_context"]
        obj._context.extend(np.asarray(c, dtype=np.float32) for c in obj._init_context)
        return obj
