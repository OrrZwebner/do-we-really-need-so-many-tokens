"""Ingestion: text cleanup, structure recovery, and chunk boundaries."""

from nelson_rag.ingest.chunker import chunk_document
from nelson_rag.ingest.loaders import Page, normalise_text, strip_html
from nelson_rag.ingest.structure import (
    DocumentText,
    build_sections,
    detect_headings,
)


def make_doc(text: str, pages: int = 1) -> DocumentText:
    if pages == 1:
        return DocumentText.from_pages([Page(1, text, "t.txt")], "t.txt")
    size = len(text) // pages + 1
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return DocumentText.from_pages(
        [Page(i, c, "t.txt") for i, c in enumerate(chunks, start=1)], "t.txt"
    )


class TestNormalisation:
    def test_rejoins_words_hyphenated_across_lines(self):
        assert "hyperkalemia" in normalise_text("hyper-\nkalemia is present")

    def test_expands_ligatures_and_smart_punctuation(self):
        out = normalise_text("the eﬀect of “therapy” — dose")
        assert "effect" in out and '"therapy"' in out and "-" in out

    def test_strips_html_to_text_with_block_breaks(self):
        out = strip_html("<h2>Treatment</h2><p>Give fluids.</p><p>Then observe.</p>")
        assert "Treatment" in out and "Give fluids." in out
        assert "<" not in out


class TestStructure:
    SAMPLE = (
        "PART XXVI\nThe Nervous System\n\n"
        "Chapter 611\nSeizures in Childhood\n\n"
        "TREATMENT\nGive midazolam 0.2 mg/kg intranasally.\n\n"
        "PROGNOSIS\nRecurrence is about 30%.\n"
    )

    def test_detects_part_chapter_and_sections(self):
        headings = detect_headings(make_doc(self.SAMPLE))
        titles = [h.title for h in headings]
        assert "The Nervous System" in titles
        assert "Seizures in Childhood" in titles
        assert "TREATMENT" in titles and "PROGNOSIS" in titles

    def test_does_not_repeat_a_title_continuation_line_as_a_section(self):
        headings = detect_headings(make_doc(self.SAMPLE))
        assert [h.title for h in headings].count("Seizures in Childhood") == 1

    def test_chapter_number_is_captured(self):
        headings = detect_headings(make_doc(self.SAMPLE))
        chapters = [h for h in headings if h.chapter_number]
        assert chapters and chapters[0].chapter_number == "611"

    def test_sections_carry_chapter_context_forward(self):
        doc = make_doc(self.SAMPLE)
        sections = build_sections(doc, detect_headings(doc))
        prognosis = [s for s in sections if s.section_title == "PROGNOSIS"]
        assert prognosis, "PROGNOSIS section not found"
        assert prognosis[0].chapter_number == "611"
        assert prognosis[0].chapter_title == "Seizures in Childhood"

    def test_citation_names_chapter_section_and_page(self):
        doc = make_doc(self.SAMPLE)
        section = [
            s for s in build_sections(doc, detect_headings(doc))
            if s.section_title == "TREATMENT"
        ][0]
        citation = section.citation("Nelson", "22nd ed.")
        assert "Ch. 611" in citation
        assert "TREATMENT" in citation
        assert "p. 1" in citation

    def test_page_numbers_track_multi_page_documents(self):
        doc = make_doc("A" * 500 + "\n\n" + "B" * 500, pages=2)
        assert doc.page_at(0) == 1
        assert doc.page_at(len(doc.text) - 1) == 2

    def test_unstructured_text_still_yields_one_section(self):
        doc = make_doc("just some prose with no headings at all, twice over. " * 5)
        sections = build_sections(doc, detect_headings(doc))
        assert len(sections) >= 1


class TestChunking:
    def _chunks(self, text, **kw):
        doc = make_doc(text)
        return chunk_document(
            doc,
            build_sections(doc, detect_headings(doc)),
            book_title="Nelson",
            book_edition="22e",
            chunk_chars=kw.get("chunk_chars", 400),
            overlap_chars=kw.get("overlap", 80),
            min_chunk_chars=kw.get("min_chars", 50),
        )

    def test_chunks_never_span_two_chapters(self):
        text = (
            "Chapter 1\nAlpha\n\n" + "Alpha content. " * 60 +
            "\n\nChapter 2\nBeta\n\n" + "Beta content. " * 60
        )
        for chunk in self._chunks(text):
            assert not ("Alpha content" in chunk.text and "Beta content" in chunk.text)

    def test_consecutive_chunks_overlap(self):
        chunks = self._chunks("Chapter 1\nAlpha\n\n" + "Sentence number one. " * 120)
        assert len(chunks) > 1
        tail = chunks[0].text[-40:]
        assert any(word in chunks[1].text for word in tail.split()[:3])

    def test_every_chunk_carries_a_citation_and_id(self):
        for chunk in self._chunks("Chapter 9\nGamma\n\n" + "Content here. " * 80):
            assert chunk.citation and chunk.chunk_id
            assert chunk.page_start >= 1

    def test_chunk_ids_are_stable_across_runs(self):
        text = "Chapter 3\nDelta\n\n" + "Repeatable content. " * 60
        assert [c.chunk_id for c in self._chunks(text)] == [
            c.chunk_id for c in self._chunks(text)
        ]

    def test_tiny_sections_are_dropped(self):
        assert self._chunks("Chapter 4\nTiny\n\nShort.", min_chars=200) == []
