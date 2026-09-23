"""Sentence Transformer embeddings for tweet windows, cached per window on disk."""
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
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device or None)
        log.info("Loaded Sentence Transformer '%s' on %s (dim=%s)",
                 model_name, self.model.device, self.model.get_sentence_embedding_dimension())
        self.cache_dir = ""
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, re.sub(r"[^\w.-]+", "_", model_name))
            os.makedirs(self.cache_dir, exist_ok=True)

    def embed_window(self, window: dict) -> np.ndarray:
        texts = prepare_texts(window["texts"])
        path = os.path.join(self.cache_dir, f"{window['window_id']}.npy") if self.cache_dir else ""
        if path and os.path.exists(path):
            emb = np.load(path)
            if len(emb) == len(texts):
                return emb
            log.warning("Cache for window %s has %s rows, expected %s; re-embedding",
                        window["window_id"], len(emb), len(texts))
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        if path:
            np.save(path, emb)
        return emb
