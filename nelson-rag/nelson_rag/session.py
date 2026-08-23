"""Wiring: turn a Settings object into a ready-to-use agent."""

from __future__ import annotations

from dataclasses import dataclass

from .agent.loop import NelsonAgent
from .config import Settings, default_settings
from .indexing.embedder import Embedder, build_embedder
from .indexing.store import IndexReader
from .retrieval.hybrid import HybridRetriever


@dataclass
class Session:
    settings: Settings
    reader: IndexReader
    embedder: Embedder
    retriever: HybridRetriever

    def agent(self, client=None) -> NelsonAgent:
        return NelsonAgent(self.settings, self.retriever, client=client)

    def close(self) -> None:
        self.reader.close()


def open_session(settings: Settings | None = None) -> Session:
    """Open the built index and prepare retrieval.

    The embedding backend recorded at build time wins over the current
    environment: querying a BGE index with hash vectors would return silent
    nonsense, which is exactly the failure mode this project must not have.
    """
    settings = settings or default_settings()
    reader = IndexReader(settings.index_dir)

    built_backend = reader.meta.get("embedding_backend")
    built_model = reader.meta.get("embedder")
    if built_backend and built_backend != settings.embedding_backend:
        settings = Settings(
            **{
                **settings.__dict__,
                "embedding_backend": built_backend,
                "local_embedding_model": built_model or settings.local_embedding_model,
            }
        )
    elif built_backend == "local" and built_model:
        settings = Settings(**{**settings.__dict__, "local_embedding_model": built_model})

    embedder = build_embedder(settings)
    if reader.meta.get("dim") and embedder.dim != reader.meta["dim"]:
        raise RuntimeError(
            f"Embedding dimension mismatch: index was built with "
            f"{reader.meta['embedder']} ({reader.meta['dim']}-d) but the current "
            f"backend gives {embedder.name} ({embedder.dim}-d). Rebuild the index "
            "or restore the original backend."
        )

    return Session(settings, reader, embedder, HybridRetriever(reader, embedder, settings))
