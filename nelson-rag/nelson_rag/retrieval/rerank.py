"""Reranking: order the fused candidates by how well they actually answer the query.

RRF is good at *recall* — pooling two different notions of relevance — but it
throws away the content of the match. Two passages that each rank #1 in one
list and #2 in the other tie exactly, and the tie gets broken by nothing
meaningful. On a query like "amoxicillin dose acute otitis media", that is the
difference between the otitis chapter and the seizures chapter.

Two rerankers:

* ``TermCoverageReranker`` (default) — free, instant, no dependencies. Scores
  a passage by how many of the query's content words it contains, weighting
  rare terms higher. Rare terms in this corpus are drug names, organisms, and
  syndrome eponyms, which is exactly what should drive the ordering.
* ``CrossEncoderReranker`` — a real cross-encoder if sentence-transformers is
  installed. Substantially better, ~50ms for 24 candidates on CPU.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from ..indexing.store import Hit

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*", re.I)

# Words too common in clinical prose to carry ranking signal.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "for", "with", "and", "or", "to", "is", "are",
    "on", "at", "by", "be", "as", "from", "that", "this", "it", "its", "what",
    "how", "when", "which", "should", "can", "does", "do", "was", "were", "has",
    "have", "not", "but", "if", "than", "then", "there", "their", "you", "your",
    "patient", "child", "children", "case", "give", "given", "use", "used",
    "treatment", "management", "dose", "dosing", "therapy",
}


class Reranker:
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        raise NotImplementedError


class NullReranker(Reranker):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        return list(hits)


def _content_terms(text: str) -> list[str]:
    return [
        w.lower()
        for w in _WORD.findall(text)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]


class TermCoverageReranker(Reranker):
    """Blend the fusion score with query-term coverage, rare terms weighted up.

    ``weight`` controls how much coverage can move a result. At the default of
    0.5 a passage containing every query term can overtake one ranked a couple
    of places above it, but cannot overturn a large fusion-score gap.
    """

    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        hits = list(hits)
        terms = set(_content_terms(query))
        if not terms or not hits:
            return hits

        # Document frequency across the candidate pool: a term appearing in
        # every candidate (e.g. "seizure" when all hits are from that chapter)
        # tells us nothing; one appearing in two of twenty tells us a lot.
        docs = [set(_content_terms(h.search_text or h.text)) for h in hits]
        n = len(docs)
        idf = {
            term: math.log(1 + n / (1 + sum(term in d for d in docs)))
            for term in terms
        }
        total_idf = sum(idf.values()) or 1.0

        best_fusion = max(h.score for h in hits) or 1.0
        scored = []
        for hit, present in zip(hits, docs):
            coverage = sum(idf[t] for t in terms if t in present) / total_idf
            blended = hit.score + self.weight * coverage * best_fusion
            scored.append((blended, hit))

        scored.sort(key=lambda pair: -pair[0])
        for blended, hit in scored:
            hit.score = blended
        return [hit for _, hit in scored]


class CrossEncoderReranker(Reranker):
    """A trained query-passage relevance model. Best quality when available."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Cross-encoder reranking needs sentence-transformers:\n"
                "    pip install sentence-transformers"
            ) from None
        self._model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        hits = list(hits)
        if not hits:
            return hits
        scores = self._model.predict([(query, h.search_text or h.text) for h in hits])
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        return sorted(hits, key=lambda h: -h.score)


def build_reranker(settings) -> Reranker:
    kind = getattr(settings, "reranker", "coverage").lower()
    if kind in {"none", "off"}:
        return NullReranker()
    if kind == "coverage":
        return TermCoverageReranker(getattr(settings, "rerank_weight", 0.5))
    if kind in {"cross-encoder", "cross_encoder", "ce"}:
        return CrossEncoderReranker(
            getattr(settings, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
    raise ValueError(
        f"Unknown reranker '{kind}'. Expected: coverage, cross-encoder, none."
    )
