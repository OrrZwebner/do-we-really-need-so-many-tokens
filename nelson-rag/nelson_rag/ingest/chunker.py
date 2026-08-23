"""Split sections into retrieval chunks.

Rules, in priority order:

1. Never split across a section boundary — a chunk belongs to exactly one
   place in the book, so its citation is unambiguous.
2. Prefer to break on a blank line, then a sentence end, then whitespace.
   Splitting mid-sentence in a dosing paragraph is how a RAG system ends up
   quoting "give 15 mg/kg" without the "every 6 hours".
3. Overlap consecutive chunks so a fact straddling a boundary is still
   retrievable whole from at least one of them.

Each chunk also carries a *context header* — "Ch. 220 Acute Otitis Media >
TREATMENT" — which is indexed but never displayed. Without it, the passage
holding the amoxicillin dose contains the word "amoxicillin" and not the words
"acute otitis media", so a query naming both scores it below the chapter's
introductory paragraph. The header is kept out of ``text`` so quotations and
citations still show the textbook's own words, unaltered.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict

from .structure import DocumentText, Section


@dataclass
class Chunk:
    chunk_id: str
    text: str            # verbatim textbook prose, shown to the reader
    search_text: str     # context header + text, indexed for retrieval
    source_file: str
    part: str | None
    chapter_number: str | None
    chapter_title: str | None
    section_title: str | None
    page_start: int
    page_end: int
    citation: str
    char_start: int
    char_end: int

    def to_row(self) -> dict:
        return asdict(self)


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _best_break(text: str, lo: int, hi: int) -> int:
    """Pick the nicest split point in ``text[lo:hi]``, preferring later breaks."""
    window = text[lo:hi]
    for pattern in (_PARAGRAPH_BREAK, _SENTENCE_END):
        matches = list(pattern.finditer(window))
        if matches:
            return lo + matches[-1].end()
    space = window.rfind(" ")
    return lo + space + 1 if space > 0 else hi


def chunk_section(
    doc: DocumentText,
    section: Section,
    book_title: str,
    book_edition: str,
    chunk_chars: int,
    overlap_chars: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    body = doc.text[section.start:section.end].strip()
    if len(body) < min_chunk_chars:
        return []

    # Where the stripped body actually starts inside the document.
    base = section.start + (section.end - section.start - len(doc.text[section.start:section.end].lstrip()))

    chunks: list[Chunk] = []
    cursor = 0
    while cursor < len(body):
        end = min(cursor + chunk_chars, len(body))
        if end < len(body):
            # Look for a clean break in the last third of the window.
            end = _best_break(body, cursor + (2 * chunk_chars) // 3, end)
        piece = body[cursor:end].strip()
        if len(piece) >= min_chunk_chars or (not chunks and piece):
            abs_start = base + cursor
            abs_end = base + end
            page_start = doc.page_at(abs_start)
            page_end = doc.page_at(max(abs_end - 1, abs_start))
            located = Section(
                start=abs_start,
                end=abs_end,
                part=section.part,
                chapter_number=section.chapter_number,
                chapter_title=section.chapter_title,
                section_title=section.section_title,
                page_start=page_start,
                page_end=page_end,
            )
            header = context_header(located)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.source_file, abs_start, piece),
                    text=piece,
                    search_text=f"{header}\n\n{piece}" if header else piece,
                    source_file=doc.source_file,
                    part=section.part,
                    chapter_number=section.chapter_number,
                    chapter_title=section.chapter_title,
                    section_title=section.section_title,
                    page_start=page_start,
                    page_end=page_end,
                    citation=located.citation(book_title, book_edition),
                    char_start=abs_start,
                    char_end=abs_end,
                )
            )
        if end >= len(body):
            break
        cursor = max(end - overlap_chars, cursor + 1)
    return chunks


def context_header(section: Section) -> str:
    """Where this passage sits in the book, as indexable words."""
    parts = []
    if section.part:
        parts.append(section.part)
    if section.chapter_number and section.chapter_title:
        parts.append(f"Chapter {section.chapter_number} {section.chapter_title}")
    elif section.chapter_title:
        parts.append(section.chapter_title)
    if section.section_title:
        parts.append(section.section_title)
    return " > ".join(parts)


def _chunk_id(source_file: str, offset: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_file}:{offset}:{text[:200]}".encode()).hexdigest()
    return digest[:16]


def chunk_document(
    doc: DocumentText,
    sections: list[Section],
    *,
    book_title: str,
    book_edition: str,
    chunk_chars: int,
    overlap_chars: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(
            chunk_section(
                doc, section, book_title, book_edition,
                chunk_chars, overlap_chars, min_chunk_chars,
            )
        )
    return chunks
