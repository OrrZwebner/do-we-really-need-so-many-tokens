"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.

Both halves earn their place on a medical corpus:

* Dense catches paraphrase — "baby won't stop crying" finding the colic
  chapter, or a Hebrew-speaking clinician's English query that uses different
  words than the textbook.
* BM25 catches exact tokens dense models blur: drug names, "G6PD", "22q11.2",
  ICD-ish strings, and numbers. Getting "amoxicillin" instead of "ampicillin"
  is not an acceptable near-miss in this domain.

RRF fuses them without needing the two score scales to be comparable, which
they are not.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..indexing.embedder import Embedder
from ..indexing.store import Hit, IndexReader
from .rerank import Reranker, build_reranker


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    dense_count: int
    lexical_count: int

    def to_context_block(self) -> str:
        if not self.hits:
            return (
                "NO PASSAGES FOUND for this query. Do not answer from memory — "
                "either search again with different wording, or tell the "
                "clinician the textbook index has nothing on this."
            )
        return "\n".join(hit.as_context(i) for i, hit in enumerate(self.hits, start=1))


class HybridRetriever:
    def __init__(
        self,
        reader: IndexReader,
        embedder: Embedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ):
        self.reader = reader
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker if reranker is not None else build_reranker(settings)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        chapter_number: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.fused_top_k

        dense = self.reader.dense_search(
            self.embedder.encode_query(query), self.settings.dense_top_k
        )
        lexical = self.reader.lexical_search(query, self.settings.lexical_top_k)

        fused = reciprocal_rank_fusion(
            [row for row, _ in dense],
            [row for row, _ in lexical],
            k=self.settings.rrf_k,
        )

        dense_rank = {row: i + 1 for i, (row, _) in enumerate(dense)}
        lexical_rank = {row: i + 1 for i, (row, _) in enumerate(lexical)}

        # Over-fetch: the reranker needs candidates to work with, and a
        # chapter filter needs even more so it does not empty the result set.
        take = max(self.settings.rerank_candidates, top_k)
        if chapter_number:
            take *= 4
        ordered_rows = [row for row, _ in fused[:take]]
        hit_map = self.reader.hits_for_rows(ordered_rows)

        hits: list[Hit] = []
        for row, score in fused[:take]:
            hit = hit_map.get(row)
            if hit is None:
                continue
            if chapter_number and str(hit.chapter_number) != str(chapter_number):
                continue
            hit.score = score
            hit.dense_rank = dense_rank.get(row)
            hit.lexical_rank = lexical_rank.get(row)
            hits.append(hit)

        hits = self.reranker.rerank(query, hits)[:top_k]
        return RetrievalResult(query, hits, len(dense), len(lexical))


def reciprocal_rank_fusion(
    *ranked_lists: list[int], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists. Score = sum over lists of 1/(k + rank).

    Ties are common — two documents each ranked #1 in one list and #2 in the
    other score identically. Breaking those by the best rank the document
    achieved anywhere beats breaking by row ID, which is arbitrary.
    """
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(
        scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0])
    )
