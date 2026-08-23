"""Turn a source file into an ordered list of pages.

Nelson ships in different shapes depending on where it was bought (Elsevier
eBook export, a scanned PDF, an EPUB from ClinicalKey, or plain text a
librarian produced). Each loader normalises its format down to the same
``Page`` records so nothing downstream has to care.

Heavy parsers are imported lazily: you only need ``pymupdf`` installed if you
actually feed the pipeline a PDF.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Page:
    """One page (PDF) or one spine document (EPUB/HTML)."""

    number: int
    text: str
    source_file: str


@dataclass
class TocEntry:
    """A table-of-contents entry, used to recover chapter structure."""

    level: int
    title: str
    page: int


class UnsupportedSourceError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', " ": " ", "​": "",
}

# A word broken across a line by a hyphen: "hyper-\nkalemia" -> "hyperkalemia".
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# Three or more newlines collapse to a paragraph break.
_EXCESS_BLANKS = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalise_text(text: str) -> str:
    """Clean up the typographic noise that PDF and EPUB extraction leaves behind."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _EXCESS_BLANKS.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def load_pdf(path: Path) -> tuple[list[Page], list[TocEntry]]:
    try:
        import pymupdf  # type: ignore
    except ImportError:  # pragma: no cover - depends on optional install
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            raise UnsupportedSourceError(
                "Reading PDFs needs PyMuPDF. Install it with: pip install pymupdf"
            ) from None

    doc = pymupdf.open(path)
    pages: list[Page] = []
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = normalise_text(raw)
        if cleaned:
            pages.append(Page(number=i, text=cleaned, source_file=path.name))

    toc = [
        TocEntry(level=int(lvl), title=str(title).strip(), page=int(pg))
        for lvl, title, pg in (doc.get_toc() or [])
        if str(title).strip() and int(pg) > 0
    ]
    doc.close()

    if not pages:
        raise UnsupportedSourceError(
            f"{path.name}: no extractable text. This is probably a scanned PDF — "
            "run OCR first (e.g. `ocrmypdf in.pdf out.pdf`) and re-ingest."
        )
    return pages, toc


# --------------------------------------------------------------------------
# EPUB
# --------------------------------------------------------------------------


def load_epub(path: Path) -> tuple[list[Page], list[TocEntry]]:
    try:
        import ebooklib  # type: ignore
        from ebooklib import epub  # type: ignore
    except ImportError:  # pragma: no cover - depends on optional install
        raise UnsupportedSourceError(
            "Reading EPUBs needs EbookLib. Install it with: pip install EbookLib"
        ) from None

    book = epub.read_epub(str(path))
    pages: list[Page] = []
    toc: list[TocEntry] = []

    for i, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), start=1):
        text = strip_html(item.get_content().decode("utf-8", errors="replace"))
        if not text:
            continue
        pages.append(Page(number=i, text=text, source_file=path.name))
        # The first heading in a spine document is its chapter title.
        heading = _first_heading(item.get_content().decode("utf-8", errors="replace"))
        if heading:
            toc.append(TocEntry(level=1, title=heading, page=i))

    if not pages:
        raise UnsupportedSourceError(f"{path.name}: EPUB contained no readable text.")
    return pages, toc


_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_BLOCK_END = re.compile(r"</(p|div|h[1-6]|li|tr|table|section)\s*>", re.I)
_HEADING = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.S | re.I)


def strip_html(markup: str) -> str:
    """Convert HTML to plain text, keeping block boundaries as newlines."""
    markup = _SCRIPT_STYLE.sub(" ", markup)
    markup = _BLOCK_END.sub("\n", markup)
    markup = markup.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = _TAG.sub(" ", markup)
    text = html.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return normalise_text(text)


def _first_heading(markup: str) -> str | None:
    match = _HEADING.search(markup)
    if not match:
        return None
    title = strip_html(match.group(1)).strip()
    return title or None


# --------------------------------------------------------------------------
# Plain text / markdown / HTML
# --------------------------------------------------------------------------

# A page break marker some exports leave behind, plus the ASCII form feed.
_PAGE_BREAK = re.compile(r"\f|^\s*(?:---\s*)?\[?page\s+(\d+)\]?\s*(?:---)?\s*$",
                         re.I | re.M)


def load_text(path: Path) -> tuple[list[Page], list[TocEntry]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm", ".xhtml"}:
        raw = strip_html(raw)

    parts = _PAGE_BREAK.split(raw)
    # re.split with a capturing group interleaves the captures; keep the text only.
    chunks = [p for i, p in enumerate(parts) if p is not None and not p.isdigit()]
    pages = []
    for i, part in enumerate(chunks, start=1):
        cleaned = normalise_text(part)
        if cleaned:
            pages.append(Page(number=i, text=cleaned, source_file=path.name))

    if not pages:
        raise UnsupportedSourceError(f"{path.name}: file is empty.")
    return pages, []


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

_LOADERS = {
    ".pdf": load_pdf,
    ".epub": load_epub,
    ".txt": load_text,
    ".md": load_text,
    ".html": load_text,
    ".htm": load_text,
    ".xhtml": load_text,
}

SUPPORTED_SUFFIXES = tuple(sorted(_LOADERS))


def load_source(path: Path) -> tuple[list[Page], list[TocEntry]]:
    """Load one source file into pages plus whatever ToC it carries."""
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedSourceError(
            f"{path.name}: unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    return loader(path)


def discover_sources(directory: Path) -> list[Path]:
    """Every ingestible file under ``directory``, in a stable order."""
    if not directory.exists():
        return []
    found = [
        p for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in _LOADERS and not p.name.startswith(".")
    ]
    return found
