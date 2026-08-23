"""Central configuration for the Nelson RAG assistant.

Every knob is overridable from the environment so the same code runs on a
laptop (local embeddings, no API cost for indexing) and on a workstation
(hosted embeddings, larger models) without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


DATA_DIR = _env_path("NELSON_DATA_DIR", PROJECT_ROOT / "data")
SOURCE_DIR = _env_path("NELSON_SOURCE_DIR", DATA_DIR / "source")
INDEX_DIR = _env_path("NELSON_INDEX_DIR", DATA_DIR / "index")

# --------------------------------------------------------------------------
# Corpus identity — surfaced in every citation so the clinician always knows
# which edition an answer came from.
# --------------------------------------------------------------------------

BOOK_TITLE = os.environ.get("NELSON_BOOK_TITLE", "Nelson Textbook of Pediatrics")
BOOK_EDITION = os.environ.get("NELSON_BOOK_EDITION", "unspecified edition")

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

# Chunks are sized in *characters*; ~4 chars/token is a good English estimate,
# so 2400 chars lands near 600 tokens — big enough to hold a full dosing
# paragraph or a diagnostic-criteria list without splitting it.
CHUNK_CHARS = int(os.environ.get("NELSON_CHUNK_CHARS", "2400"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("NELSON_CHUNK_OVERLAP_CHARS", "300"))
MIN_CHUNK_CHARS = int(os.environ.get("NELSON_MIN_CHUNK_CHARS", "200"))

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

# "local"  -> sentence-transformers, runs offline, nothing leaves the machine.
# "voyage" -> Voyage AI hosted embeddings (higher quality, sends text out).
# "hash"   -> deterministic bag-of-words hashing; no model download. Used by
#             the test suite and as a smoke-test fallback. Not for real use.
EMBEDDING_BACKEND = os.environ.get("NELSON_EMBEDDING_BACKEND", "local")
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "NELSON_LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
)
VOYAGE_EMBEDDING_MODEL = os.environ.get("NELSON_VOYAGE_MODEL", "voyage-3")
HASH_EMBEDDING_DIM = int(os.environ.get("NELSON_HASH_DIM", "512"))
EMBED_BATCH_SIZE = int(os.environ.get("NELSON_EMBED_BATCH", "64"))

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

DENSE_TOP_K = int(os.environ.get("NELSON_DENSE_TOP_K", "30"))
LEXICAL_TOP_K = int(os.environ.get("NELSON_LEXICAL_TOP_K", "30"))
FUSED_TOP_K = int(os.environ.get("NELSON_FUSED_TOP_K", "8"))
RRF_K = int(os.environ.get("NELSON_RRF_K", "60"))

# How many fused candidates to rerank before truncating to FUSED_TOP_K.
RERANK_CANDIDATES = int(os.environ.get("NELSON_RERANK_CANDIDATES", "24"))
# "coverage" (free, default) | "cross-encoder" (better, needs a model) | "none"
RERANKER = os.environ.get("NELSON_RERANKER", "coverage")
RERANK_WEIGHT = float(os.environ.get("NELSON_RERANK_WEIGHT", "0.5"))
CROSS_ENCODER_MODEL = os.environ.get(
    "NELSON_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --------------------------------------------------------------------------
# Model / agent
# --------------------------------------------------------------------------

ANSWER_MODEL = os.environ.get("NELSON_ANSWER_MODEL", "claude-opus-5")
MAX_TOKENS = int(os.environ.get("NELSON_MAX_TOKENS", "16000"))
EFFORT = os.environ.get("NELSON_EFFORT", "high")  # low|medium|high|xhigh|max
MAX_TOOL_ROUNDS = int(os.environ.get("NELSON_MAX_TOOL_ROUNDS", "8"))


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the configuration, passed explicitly where used."""

    index_dir: Path = INDEX_DIR
    source_dir: Path = SOURCE_DIR
    book_title: str = BOOK_TITLE
    book_edition: str = BOOK_EDITION
    chunk_chars: int = CHUNK_CHARS
    chunk_overlap_chars: int = CHUNK_OVERLAP_CHARS
    min_chunk_chars: int = MIN_CHUNK_CHARS
    embedding_backend: str = EMBEDDING_BACKEND
    local_embedding_model: str = LOCAL_EMBEDDING_MODEL
    voyage_embedding_model: str = VOYAGE_EMBEDDING_MODEL
    hash_embedding_dim: int = HASH_EMBEDDING_DIM
    embed_batch_size: int = EMBED_BATCH_SIZE
    dense_top_k: int = DENSE_TOP_K
    lexical_top_k: int = LEXICAL_TOP_K
    fused_top_k: int = FUSED_TOP_K
    rrf_k: int = RRF_K
    rerank_candidates: int = RERANK_CANDIDATES
    reranker: str = RERANKER
    rerank_weight: float = RERANK_WEIGHT
    cross_encoder_model: str = CROSS_ENCODER_MODEL
    answer_model: str = ANSWER_MODEL
    max_tokens: int = MAX_TOKENS
    effort: str = EFFORT
    max_tool_rounds: int = MAX_TOOL_ROUNDS
    extra: dict = field(default_factory=dict)


def default_settings(**overrides) -> Settings:
    return Settings(**overrides)
