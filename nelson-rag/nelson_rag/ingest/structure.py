"""Recover Nelson's part / chapter / section hierarchy from flat page text.

Why this matters: a retrieval hit is only clinically useful if the reader can
see *where* it came from. "Nelson, Ch. 573 'Hypoglycemia in Infants and
Children' > Treatment, p. 3421" is checkable; "chunk #18422" is not. So the
ingest pipeline spends real effort reconstructing structure, and every chunk
carries it.

Two sources of structure, in order of trust:

1. The embedded table of contents (PDF outline / EPUB nav). Publisher-produced
   and accurate when present.
2. Heading regexes over the text itself. Nelson's headings are extremely
   regular ("PART XVII", "Chapter 573"), which makes this reliable even for
   exports that dropped the outline.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from .loaders import Page, TocEntry

# --------------------------------------------------------------------------
# A whole book flattened into one addressable string
# --------------------------------------------------------------------------


@dataclass
class DocumentText:
    """Concatenated page text plus an offset->page index."""

    text: str
    source_file: str
    _offsets: list[int] = field(default_factory=list)
    _pages: list[int] = field(default_factory=list)

    @classmethod
    def from_pages(cls, pages: list[Page], source_file: str) -> "DocumentText":
        parts: list[str] = []
        offsets: list[int] = []
        numbers: list[int] = []
        cursor = 0
        for page in pages:
            offsets.append(cursor)
            numbers.append(page.number)
            block = page.text + "\n\n"
            parts.append(block)
            cursor += len(block)
        return cls("".join(parts), source_file, offsets, numbers)

    def page_at(self, offset: int) -> int:
        """Page number containing ``offset`` (1 when the map is empty)."""
        if not self._offsets:
            return 1
        idx = bisect.bisect_right(self._offsets, offset) - 1
        return self._pages[max(idx, 0)]

    def offset_of_page(self, page: int) -> int:
        try:
            return self._offsets[self._pages.index(page)]
        except ValueError:
            return 0


# --------------------------------------------------------------------------
# Heading detection
# --------------------------------------------------------------------------

LEVEL_PART = 0
LEVEL_CHAPTER = 1
LEVEL_SECTION = 2


@dataclass
class Heading:
    offset: int
    level: int
    title: str
    chapter_number: str | None = None


# "PART XVII  The Endocrine System" (title may sit on the next line)
_PART_RE = re.compile(
    r"^[ \t]*PART[ \t]+([IVXLCDM]+|\d+)[ \t]*[:.—-]?[ \t]*(.*)$", re.M
)
# "Chapter 573  Hypoglycemia" / "CHAPTER 573" / "573  Hypoglycemia" is too loose,
# so we require the literal word.
_CHAPTER_RE = re.compile(
    r"^[ \t]*(?:Chapter|CHAPTER)[ \t]+(\d+(?:\.\d+)?)[ \t]*[:.—-]?[ \t]*(.*)$", re.M
)

# Section headings inside a chapter: a short standalone line, Title Case or
# ALL CAPS, no sentence-ending punctuation. Nelson's are things like
# "CLINICAL MANIFESTATIONS", "Treatment", "Differential Diagnosis".
_SECTION_RE = re.compile(r"^[ \t]*([A-Z][^\n]{2,70})[ \t]*$", re.M)

_SECTION_STOPWORDS = {
    "figure", "table", "box", "fig", "see", "chapter", "part", "references",
    "bibliography", "keywords", "downloaded", "copyright",
}

# Lines that are really running heads / page furniture, not section titles.
_FURNITURE_RE = re.compile(
    r"^(?:\d+\s+)?(?:part|chapter|section|unit)\b|^\s*\d+\s*$|elsevier|nelson textbook",
    re.I,
)


def _looks_like_section(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 70):
        return False
    if stripped[-1] in ".,;:?!)":
        return False
    if _FURNITURE_RE.search(stripped):
        return False
    words = stripped.split()
    if not (1 <= len(words) <= 9):
        return False
    if words[0].lower().strip(":") in _SECTION_STOPWORDS:
        return False
    if sum(ch.isdigit() for ch in stripped) > len(stripped) / 3:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    # Either SHOUTED or Title Cased — both are Nelson heading styles.
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    title_cased = all(w[0].isupper() or not w[0].isalpha() or w.lower() in
                      {"and", "of", "the", "in", "for", "with", "to", "or", "a", "an"}
                      for w in words)
    return upper_ratio > 0.85 or title_cased


def _title_after(text: str, end_of_line: int, fallback: str) -> tuple[str, int]:
    """A part/chapter number often sits alone; the title is then the next line.

    Returns the title and the offset up to which text was consumed, so the
    section pass can skip a continuation line instead of re-reporting it as a
    heading of its own.
    """
    if fallback.strip():
        return fallback.strip(), end_of_line
    nxt = text.find("\n", end_of_line + 1)
    stop = nxt if nxt != -1 else min(end_of_line + 120, len(text))
    candidate = text[end_of_line + 1: stop]
    return (candidate.strip() or "(untitled)"), stop


def detect_headings(doc: DocumentText, toc: list[TocEntry] | None = None) -> list[Heading]:
    """All part/chapter/section headings in the document, ordered by offset."""
    headings: list[Heading] = []
    # Character ranges already claimed by a part/chapter heading (including any
    # continuation line holding its title).
    claimed: list[tuple[int, int]] = []

    for match in _PART_RE.finditer(doc.text):
        title, consumed = _title_after(doc.text, match.end(), match.group(2))
        headings.append(Heading(offset=match.start(), level=LEVEL_PART, title=title))
        claimed.append((match.start(), consumed))

    for match in _CHAPTER_RE.finditer(doc.text):
        title, consumed = _title_after(doc.text, match.end(), match.group(2))
        headings.append(
            Heading(
                offset=match.start(),
                level=LEVEL_CHAPTER,
                title=title,
                chapter_number=match.group(1),
            )
        )
        claimed.append((match.start(), consumed))

    # Fall back to the publisher ToC for chapters the regex missed.
    if toc:
        known = {h.offset for h in headings if h.level == LEVEL_CHAPTER}
        for entry in toc:
            offset = doc.offset_of_page(entry.page)
            if any(abs(offset - k) < 400 for k in known):
                continue
            number = None
            m = re.match(r"(?:chapter\s+)?(\d+(?:\.\d+)?)[\s:.—-]+(.*)", entry.title, re.I)
            title = entry.title
            if m:
                number, title = m.group(1), m.group(2) or entry.title
            headings.append(
                Heading(
                    offset=offset,
                    level=LEVEL_CHAPTER if entry.level <= 1 else LEVEL_SECTION,
                    title=title.strip(),
                    chapter_number=number,
                )
            )

    coarse = {h.offset for h in headings}
    for match in _SECTION_RE.finditer(doc.text):
        if match.start() in coarse:
            continue
        if any(lo <= match.start() <= hi for lo, hi in claimed):
            continue
        line = match.group(1)
        if _looks_like_section(line):
            headings.append(
                Heading(offset=match.start(), level=LEVEL_SECTION, title=line.strip())
            )

    headings.sort(key=lambda h: (h.offset, h.level))
    # Drop duplicate headings landing on the same offset (ToC + regex agreeing).
    deduped: list[Heading] = []
    for h in headings:
        if deduped and h.offset == deduped[-1].offset:
            continue
        deduped.append(h)
    return deduped


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass
class Section:
    """A contiguous span of text with its full structural address."""

    start: int
    end: int
    part: str | None
    chapter_number: str | None
    chapter_title: str | None
    section_title: str | None
    page_start: int
    page_end: int

    def citation(self, book: str, edition: str) -> str:
        bits = [book]
        if edition and edition != "unspecified edition":
            bits.append(edition)
        if self.chapter_number:
            chapter = f"Ch. {self.chapter_number}"
            if self.chapter_title:
                chapter += f" — {self.chapter_title}"
            bits.append(chapter)
        elif self.chapter_title:
            bits.append(self.chapter_title)
        if self.section_title:
            bits.append(self.section_title)
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        bits.append(pages)
        return ", ".join(bits)


def build_sections(doc: DocumentText, headings: list[Heading]) -> list[Section]:
    """Slice the document into sections, carrying part/chapter context forward."""
    if not headings:
        return [
            Section(
                start=0,
                end=len(doc.text),
                part=None,
                chapter_number=None,
                chapter_title=None,
                section_title=None,
                page_start=doc.page_at(0),
                page_end=doc.page_at(max(len(doc.text) - 1, 0)),
            )
        ]

    sections: list[Section] = []
    part: str | None = None
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_title: str | None = None

    # Text before the first heading (front matter) is still worth indexing.
    boundaries = [h.offset for h in headings] + [len(doc.text)]
    if headings[0].offset > 0:
        sections.append(
            Section(0, headings[0].offset, None, None, None, None,
                    doc.page_at(0), doc.page_at(headings[0].offset - 1))
        )

    for i, heading in enumerate(headings):
        if heading.level == LEVEL_PART:
            part, chapter_number, chapter_title, section_title = heading.title, None, None, None
        elif heading.level == LEVEL_CHAPTER:
            chapter_number, chapter_title, section_title = (
                heading.chapter_number, heading.title, None
            )
        else:
            section_title = heading.title

        start, end = heading.offset, boundaries[i + 1]
        if end <= start:
            continue
        sections.append(
            Section(
                start=start,
                end=end,
                part=part,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                section_title=section_title,
                page_start=doc.page_at(start),
                page_end=doc.page_at(max(end - 1, start)),
            )
        )
    return sections
