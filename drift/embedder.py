"""Sentence Transformer embeddings for tweet windows, cached per window on disk."""
import hashlib
import logging
import os
import re

import numpy as np

log = logging.getLogger("driftlens.embedder")


def prepare_texts(texts):
    """Drop U+FFFD left by encoding damage in the source CSVs; Task 2 already did the real cleaning."""
    out = [t.replace("�", "").strip() for t in texts]
    return [t for t in out if t]


class Embedder:
    def __init__(self, model_name: str, cache_dir: str = "", batch_size: int = 256, device: str = ""):
        import torch
        from sentence_transformers import SentenceTransformer

        if not device:
            if torch.cuda.is_available():
                try:
                    _ = torch.zeros(1, device="cuda")
                    device = "cuda"
                except Exception as ex:
                    log.warning("CUDA is detected but unsupported by installed PyTorch (%s); falling back to CPU", ex)
                    device = "cpu"
            else:
                device = "cpu"

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        log.info("Loaded Sentence Transformer '%s' on %s (dim=%s)",
                 model_name, self.model.device, self.model.get_sentence_embedding_dimension())
        self.cache_dir = ""
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, re.sub(r"[^\w.-]+", "_", model_name))
            os.makedirs(self.cache_dir, exist_ok=True)

    def embed_window(self, window: dict) -> np.ndarray:
        texts = prepare_texts(window["texts"])
        path = os.path.join(self.cache_dir, f"{window['window_id']}.npy") if self.cache_dir else ""
        # The cache is keyed by window_id, so a sidecar hash of the texts catches a window whose tweets
        # changed but whose id and size did not (e.g. regenerated experiment scenarios).
        digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()
        if path and os.path.exists(path):
            emb = np.load(path)
            if len(emb) == len(texts) and _read(path + ".sha1") == digest:
                return emb
            log.warning("Cache for window %s is stale (texts changed or no hash); re-embedding", window["window_id"])
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, emb)
            with open(path + ".sha1", "w") as f:
                f.write(digest)
        return emb


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""
