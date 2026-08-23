"""On-disk index: SQLite for metadata + BM25, a NumPy array for the vectors.

Two files under the index directory:

    index.sqlite   chunk rows, structural metadata, and an FTS5 BM25 index
    vectors.npy    float32 [n_chunks, dim], row i <-> chunks.vec_row = i

Deliberately boring. A textbook is ~30-60k chunks; a flat NumPy dot product
over that is a few milliseconds and needs no native index library, no server,
and no version-skew headaches. FAISS is used automatically if it happens to be
installed, but is never required.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    vec_row        INTEGER PRIMARY KEY,
    chunk_id       TEXT NOT NULL UNIQUE,
    text           TEXT NOT NULL,
    search_text    TEXT NOT NULL,
    source_file    TEXT NOT NULL,
    part           TEXT,
    chapter_number TEXT,
    chapter_title  TEXT,
    section_title  TEXT,
    page_start     INTEGER,
    page_end       INTEGER,
    citation       TEXT NOT NULL,
    char_start     INTEGER,
    char_end       INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter_number);
CREATE INDEX IF NOT EXISTS idx_chunks_source  ON chunks(source_file);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    search_text,
    chapter_title,
    section_title,
    content='chunks',
    content_rowid='vec_row',
    tokenize='porter unicode61'
);
"""


@dataclass
class Hit:
    """A retrieved chunk plus the scores that surfaced it."""

    chunk_id: str
    text: str
    citation: str
    chapter_number: str | None
    chapter_title: str | None
    section_title: str | None
    page_start: int
    page_end: int
    source_file: str
    # Context header + text: what retrieval matched against. Never displayed.
    search_text: str = ""
    score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None

    def as_context(self, index: int) -> str:
        """Render for the model's tool result, with an explicit citation tag."""
        return (
            f"[{index}] {self.citation}\n"
            f"----------------------------------------\n"
            f"{self.text}\n"
        )


def _row_to_hit(row: sqlite3.Row) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"],
        text=row["text"],
        citation=row["citation"],
        chapter_number=row["chapter_number"],
        chapter_title=row["chapter_title"],
        section_title=row["section_title"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        source_file=row["source_file"],
        search_text=row["search_text"],
    )


class IndexWriter:
    """Builds an index directory from scratch."""

    def __init__(self, index_dir: Path, dim: int):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = index_dir / "index.sqlite"
        self.vec_path = index_dir / "vectors.npy"
        if self.db_path.exists():
            self.db_path.unlink()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self._next_row = 0

    def add(self, chunks: Sequence, vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        rows = []
        for chunk in chunks:
            rows.append(
                (
                    self._next_row, chunk.chunk_id, chunk.text, chunk.search_text,
                    chunk.source_file,
                    chunk.part, chunk.chapter_number, chunk.chapter_title,
                    chunk.section_title, chunk.page_start, chunk.page_end,
                    chunk.citation, chunk.char_start, chunk.char_end,
                )
            )
            self._next_row += 1
        self.conn.executemany(
            "INSERT OR IGNORE INTO chunks (vec_row, chunk_id, text, search_text,"
            " source_file, part, chapter_number, chapter_title, section_title,"
            " page_start, page_end, citation, char_start, char_end)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._vectors.append(np.asarray(vectors, dtype=np.float32))

    def finalise(self, meta: dict) -> None:
        # Rebuild the FTS index from the content table in one pass — much
        # faster than maintaining it row by row during a full ingest.
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self.conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in meta.items()],
        )
        self.conn.commit()
        matrix = (
            np.vstack(self._vectors)
            if self._vectors
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        np.save(self.vec_path, matrix)
        self.conn.close()


# FTS5 treats these as query syntax; a clinical question full of them would
# otherwise raise "fts5: syntax error".
_FTS_SPECIAL = re.compile(r'[^\w\s]', re.UNICODE)


def _fts_query(text: str) -> str:
    """Turn free text into a safe OR-query of quoted terms."""
    terms = [t for t in _FTS_SPECIAL.sub(" ", text).split() if len(t) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms[:40])


class IndexReader:
    """Read-only access to a built index."""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        db_path = self.index_dir / "index.sqlite"
        vec_path = self.index_dir / "vectors.npy"
        if not db_path.exists():
            raise FileNotFoundError(
                f"No index at {self.index_dir}. Build one first:\n"
                "    python -m nelson_rag.cli ingest --source data/source"
            )
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.vectors = (
            np.load(vec_path, mmap_mode="r")
            if vec_path.exists()
            else np.zeros((0, 1), dtype=np.float32)
        )
        self.meta = {
            row["key"]: json.loads(row["value"])
            for row in self.conn.execute("SELECT key, value FROM meta")
        }

    # -- statistics --------------------------------------------------------

    def chunk_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # -- search ------------------------------------------------------------

    def dense_search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.vectors.shape[0] == 0:
            return []
        scores = np.asarray(self.vectors) @ np.asarray(query_vector, dtype=np.float32)
        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def lexical_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        match = _fts_query(query)
        if not match:
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts, 1.0, 2.0, 1.5) AS score"
                " FROM chunks_fts WHERE chunks_fts MATCH ?"
                " ORDER BY score LIMIT ?",
                (match, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() is negative-is-better in SQLite; flip it so bigger is better.
        return [(int(r["rowid"]), -float(r["score"])) for r in rows]

    def hits_for_rows(self, rows: Iterable[int]) -> dict[int, Hit]:
        rows = list(rows)
        if not rows:
            return {}
        placeholders = ",".join("?" * len(rows))
        found = self.conn.execute(
            f"SELECT * FROM chunks WHERE vec_row IN ({placeholders})", rows
        ).fetchall()
        return {int(r["vec_row"]): _row_to_hit(r) for r in found}

    # -- structural browsing ----------------------------------------------

    def list_chapters(self, contains: str | None = None, limit: int = 60) -> list[dict]:
        sql = (
            "SELECT chapter_number, chapter_title, MIN(page_start) AS page,"
            " COUNT(*) AS chunks FROM chunks WHERE chapter_title IS NOT NULL"
        )
        params: list = []
        if contains:
            sql += " AND (chapter_title LIKE ? OR chapter_number = ?)"
            params += [f"%{contains}%", contains]
        sql += (
            " GROUP BY chapter_number, chapter_title"
            " ORDER BY CAST(chapter_number AS INTEGER), page LIMIT ?"
        )
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params)]

    def chapter_chunks(self, chapter_number: str, limit: int = 40) -> list[Hit]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE chapter_number = ?"
            " ORDER BY char_start LIMIT ?",
            (str(chapter_number), limit),
        ).fetchall()
        return [_row_to_hit(r) for r in rows]

    def chapter_outline(self, chapter_number: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT section_title FROM chunks"
            " WHERE chapter_number = ? AND section_title IS NOT NULL"
            " GROUP BY section_title ORDER BY MIN(char_start)",
            (str(chapter_number),),
        ).fetchall()
        return [r["section_title"] for r in rows]

    def close(self) -> None:
        self.conn.close()
