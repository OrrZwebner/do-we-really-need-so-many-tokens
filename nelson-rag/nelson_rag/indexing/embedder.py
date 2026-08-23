"""Pluggable text embedding backends.

Default is **local**: the textbook never leaves the machine, indexing is free,
and it works on a plane. Voyage is available for people who want the extra
retrieval quality and are comfortable sending passages to a third party.

All backends return L2-normalised float32 vectors, so cosine similarity is a
plain dot product.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

import numpy as np


class Embedder:
    """Interface every backend implements."""

    name: str = "base"
    dim: int = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


# --------------------------------------------------------------------------
# Hash backend — no downloads, deterministic, used by tests
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder(Embedder):
    """Hashed bag-of-words with sublinear term weighting.

    Deliberately simple. It exists so the pipeline is runnable and testable
    with zero model downloads — it is *not* a substitute for a real embedding
    model on a live corpus, and ``build_index`` warns when it is in use.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            tokens = _TOKEN.findall(text.lower())
            for token in tokens:
                bucket = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % self.dim
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            # Bigrams give the vector a little word-order sensitivity.
            for a, b in zip(tokens, tokens[1:]):
                bucket = int(hashlib.md5(f"{a}_{b}".encode()).hexdigest()[:8], 16) % self.dim
                counts[bucket] = counts.get(bucket, 0.0) + 0.5
            for bucket, count in counts.items():
                out[row, bucket] = 1.0 + np.log(count)
        return _l2_normalise(out)


# --------------------------------------------------------------------------
# Local sentence-transformers backend
# --------------------------------------------------------------------------

# BGE-family models are trained with an instruction prefix on the *query*
# side only; adding it lifts recall noticeably.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Local embeddings need sentence-transformers:\n"
                "    pip install sentence-transformers\n"
                "Or switch backend: NELSON_EMBEDDING_BACKEND=voyage (hosted) "
                "or =hash (offline smoke test)."
            ) from None
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._batch_size = batch_size
        self._uses_bge_prefix = "bge" in model_name.lower()

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        if self._uses_bge_prefix:
            text = _BGE_QUERY_PREFIX + text
        return self.encode_documents([text])[0]


# --------------------------------------------------------------------------
# Voyage AI hosted backend
# --------------------------------------------------------------------------


class VoyageEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 64):
        try:
            import voyageai  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Voyage embeddings need the voyageai package:\n"
                "    pip install voyageai\n"
                "and a VOYAGE_API_KEY in the environment."
            ) from None
        self._client = voyageai.Client()
        self.name = model_name
        self._batch_size = min(batch_size, 128)
        self.dim = len(self._embed(["dimension probe"], "document")[0])

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        return self._client.embed(
            list(texts), model=self.name, input_type=input_type
        ).embeddings

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            rows.extend(self._embed(texts[start:start + self._batch_size], "document"))
        return _l2_normalise(np.asarray(rows, dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        vec = np.asarray(self._embed([text], "query"), dtype=np.float32)
        return _l2_normalise(vec)[0]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_embedder(settings) -> Embedder:
    backend = settings.embedding_backend.lower()
    if backend == "hash":
        return HashEmbedder(settings.hash_embedding_dim)
    if backend == "local":
        return LocalEmbedder(settings.local_embedding_model, settings.embed_batch_size)
    if backend == "voyage":
        return VoyageEmbedder(settings.voyage_embedding_model, settings.embed_batch_size)
    raise ValueError(
        f"Unknown embedding backend '{settings.embedding_backend}'. "
        "Expected one of: local, voyage, hash."
    )
