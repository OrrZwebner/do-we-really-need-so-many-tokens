"""Build the searchable index from source files."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..config import Settings
from ..ingest.chunker import Chunk, chunk_document
from ..ingest.loaders import UnsupportedSourceError, discover_sources, load_source
from ..ingest.structure import DocumentText, build_sections, detect_headings
from .embedder import Embedder, build_embedder
from .store import IndexWriter

Progress = Callable[[str], None]


@dataclass
class BuildReport:
    files: int
    chunks: int
    chapters: int
    seconds: float
    embedder: str
    dim: int
    skipped: list[tuple[str, str]]

    def summary(self) -> str:
        lines = [
            f"Indexed {self.chunks:,} chunks from {self.files} file(s) "
            f"across {self.chapters:,} chapter(s) in {self.seconds:.1f}s",
            f"Embeddings: {self.embedder} ({self.dim}-d)",
        ]
        for name, reason in self.skipped:
            lines.append(f"  skipped {name}: {reason}")
        return "\n".join(lines)


def build_index(
    settings: Settings,
    source_dir: Path | None = None,
    progress: Progress = lambda _msg: None,
) -> BuildReport:
    started = time.time()
    source_dir = Path(source_dir or settings.source_dir)
    files = discover_sources(source_dir)
    if not files:
        raise FileNotFoundError(
            f"No ingestible files under {source_dir}.\n"
            "Put your Nelson export there (.pdf, .epub, .html, .txt, .md) and retry."
        )

    progress(f"Loading embedder ({settings.embedding_backend})...")
    embedder: Embedder = build_embedder(settings)
    if settings.embedding_backend.lower() == "hash":
        progress(
            "  WARNING: the 'hash' backend is a dependency-free stand-in for "
            "smoke tests.\n           Retrieval quality will be poor. Install "
            "sentence-transformers and\n           set NELSON_EMBEDDING_BACKEND=local "
            "for real use."
        )

    writer = IndexWriter(settings.index_dir, embedder.dim)
    skipped: list[tuple[str, str]] = []
    total_chunks = 0
    chapters: set[tuple[str | None, str | None]] = set()

    for path in files:
        progress(f"Reading {path.name}...")
        try:
            pages, toc = load_source(path)
        except UnsupportedSourceError as exc:
            skipped.append((path.name, str(exc)))
            progress(f"  skipped: {exc}")
            continue

        doc = DocumentText.from_pages(pages, path.name)
        headings = detect_headings(doc, toc)
        sections = build_sections(doc, headings)
        chunks = chunk_document(
            doc,
            sections,
            book_title=settings.book_title,
            book_edition=settings.book_edition,
            chunk_chars=settings.chunk_chars,
            overlap_chars=settings.chunk_overlap_chars,
            min_chunk_chars=settings.min_chunk_chars,
        )
        progress(
            f"  {len(pages):,} pages -> {len(headings):,} headings "
            f"-> {len(chunks):,} chunks"
        )
        if not chunks:
            skipped.append((path.name, "produced no chunks above the minimum size"))
            continue

        for batch in _batched(chunks, settings.embed_batch_size):
            vectors = embedder.encode_documents([c.search_text for c in batch])
            writer.add(batch, vectors)
            total_chunks += len(batch)
            progress(f"    embedded {total_chunks:,} chunks", )

        chapters.update((c.chapter_number, c.chapter_title) for c in chunks)

    writer.finalise(
        {
            "book_title": settings.book_title,
            "book_edition": settings.book_edition,
            "embedder": embedder.name,
            "embedding_backend": settings.embedding_backend,
            "dim": embedder.dim,
            "chunk_chars": settings.chunk_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_files": [p.name for p in files],
        }
    )

    return BuildReport(
        files=len(files) - len(skipped),
        chunks=total_chunks,
        chapters=len(chapters),
        seconds=time.time() - started,
        embedder=embedder.name,
        dim=embedder.dim,
        skipped=skipped,
    )


def _batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
