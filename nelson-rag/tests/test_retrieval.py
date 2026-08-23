"""Indexing and retrieval against the fixture corpus."""

import numpy as np
import pytest

from nelson_rag.indexing.embedder import HashEmbedder
from nelson_rag.indexing.store import _fts_query
from nelson_rag.retrieval.hybrid import reciprocal_rank_fusion
from nelson_rag.retrieval.rerank import TermCoverageReranker


class TestEmbedder:
    def test_vectors_are_unit_length(self):
        vectors = HashEmbedder(128).encode_documents(["fever", "seizure in a child"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_related_text_scores_above_unrelated(self):
        e = HashEmbedder(256)
        v = e.encode_documents(
            ["febrile seizure management", "febrile seizure treatment", "aortic valve"]
        )
        assert float(v[0] @ v[1]) > float(v[0] @ v[2])

    def test_encoding_is_deterministic(self):
        a = HashEmbedder(64).encode_documents(["otitis media"])
        b = HashEmbedder(64).encode_documents(["otitis media"])
        assert np.array_equal(a, b)


class TestFtsSanitisation:
    @pytest.mark.parametrize(
        "query",
        [
            'fever >38.5°C in a 3-month-old: work-up?',
            'AND OR NOT "unbalanced',
            "amoxicillin/clavulanate (high-dose)",
            "22q11.2 deletion",
        ],
    )
    def test_clinical_punctuation_never_breaks_fts(self, session, query):
        # The point is that these do not raise a MATCH syntax error.
        session.reader.lexical_search(query, 5)

    def test_query_with_no_usable_terms_is_empty(self):
        assert _fts_query("?? -- !!") == ""


class TestFusion:
    def test_documents_ranked_by_both_lists_win(self):
        fused = dict(reciprocal_rank_fusion([1, 2, 3], [3, 4, 5], k=10))
        assert fused[3] > fused[1]

    def test_ties_break_on_best_rank_not_id(self):
        ranked = reciprocal_rank_fusion([9, 1], [1, 9], k=10)
        # Both score identically; the one ranked #1 first in list order wins.
        assert ranked[0][1] == pytest.approx(ranked[1][1])


class TestReranker:
    def test_passage_containing_the_rare_query_term_is_promoted(self, session):
        hits = session.retriever.search("amoxicillin dose acute otitis media", top_k=3).hits
        assert "amoxicillin" in hits[0].text.lower()

    def test_reranker_returns_every_candidate(self):
        from nelson_rag.indexing.store import Hit

        hits = [
            Hit(f"c{i}", f"text about topic {i}", f"cite {i}", None, None, None, 1, 1, "f")
            for i in range(5)
        ]
        assert len(TermCoverageReranker().rerank("topic 3", hits)) == 5


class TestIndex:
    def test_index_reports_what_was_built(self, built_index):
        assert built_index.chunks > 0
        assert built_index.chapters >= 3

    def test_metadata_records_the_embedder_used(self, session):
        assert session.reader.meta["embedding_backend"] == "hash"
        assert session.reader.meta["dim"] == session.embedder.dim

    def test_chapters_are_listed_with_titles(self, session):
        titles = {c["chapter_title"] for c in session.reader.list_chapters()}
        assert "Acute Otitis Media" in titles

    def test_chapter_filter_restricts_results(self, session):
        hits = session.retriever.search("treatment", top_k=5, chapter_number="220").hits
        assert hits and all(h.chapter_number == "220" for h in hits)

    def test_chapter_outline_lists_sections_in_order(self, session):
        outline = session.reader.chapter_outline("611")
        assert "TREATMENT" in outline
        assert outline.index("DEFINITION") < outline.index("TREATMENT")

    def test_exact_drug_name_is_findable(self, session):
        hits = session.retriever.search("ceftriaxone empiric meningitis", top_k=3).hits
        assert any("ceftriaxone" in h.text.lower() for h in hits)

    def test_every_hit_carries_a_citation(self, session):
        for hit in session.retriever.search("febrile seizure", top_k=5).hits:
            assert hit.citation and "Nelson" in hit.citation

    def test_reopening_a_mismatched_backend_is_refused(self, settings):
        from dataclasses import replace

        from nelson_rag.session import open_session

        # A different hash dimension is a stand-in for "wrong embedder".
        mismatched = replace(settings, hash_embedding_dim=64)
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            open_session(mismatched)
