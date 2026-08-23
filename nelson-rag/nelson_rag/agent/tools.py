"""Tool definitions and dispatch for the Nelson agent.

Three tools, matching the three ways a clinician actually uses a textbook:
look something up, find the right chapter, and read that chapter properly.

Schemas are `strict` so tool inputs are guaranteed to validate — a search with
a malformed `chapter_number` should never reach the retriever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..indexing.store import Hit
from ..retrieval.hybrid import HybridRetriever

SEARCH_TOOL = {
    "name": "search_nelson",
    "description": (
        "Search the full text of the pediatrics textbook and return the most "
        "relevant passages with their chapter, section, and page. This is the "
        "primary tool: call it before answering any clinical question. Query in "
        "English using clinical vocabulary. Prefer several focused searches over "
        "one broad one."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "English clinical search query, e.g. 'simple febrile seizure "
                    "management' or 'amoxicillin dose acute otitis media'."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "How many passages to return (1-20). Default 8.",
            },
            "chapter_number": {
                "type": ["string", "null"],
                "description": (
                    "Restrict the search to one chapter number, e.g. '573'. Use "
                    "null for a whole-book search."
                ),
            },
        },
        "required": ["query", "top_k", "chapter_number"],
        "additionalProperties": False,
    },
}

LIST_CHAPTERS_TOOL = {
    "name": "list_nelson_chapters",
    "description": (
        "List chapters in the textbook, optionally filtered by a substring of the "
        "chapter title. Use this to orient yourself when a search misses, or when "
        "the clinician asks what the book covers on a topic."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "contains": {
                "type": ["string", "null"],
                "description": (
                    "Case-insensitive substring of the chapter title, e.g. "
                    "'seizure'. Null lists chapters from the start."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum chapters to return (1-100). Default 40.",
            },
        },
        "required": ["contains", "limit"],
        "additionalProperties": False,
    },
}

READ_CHAPTER_TOOL = {
    "name": "read_nelson_chapter",
    "description": (
        "Read a chapter in reading order: its section outline plus consecutive "
        "passages. Use this when the clinician wants a full workup, a complete "
        "differential, or a whole management approach — cases where scattered "
        "search hits are not enough."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "string",
                "description": "Chapter number as shown by list_nelson_chapters, e.g. '573'.",
            },
            "max_passages": {
                "type": "integer",
                "description": "Maximum passages to return (1-25). Default 12.",
            },
        },
        "required": ["chapter_number", "max_passages"],
        "additionalProperties": False,
    },
}

TOOLS = [SEARCH_TOOL, LIST_CHAPTERS_TOOL, READ_CHAPTER_TOOL]


def _clamp(value, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class ToolRunner:
    """Executes the agent's tool calls against the index.

    Also records every passage it hands back, so the answer can be checked
    against what was actually retrieved (see ``verify.py``) and so the CLI can
    show the clinician the underlying sources.
    """

    retriever: HybridRetriever
    retrieved: list[Hit] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def run(self, name: str, payload: dict) -> str:
        self.calls.append({"tool": name, "input": payload})
        handler = {
            "search_nelson": self._search,
            "list_nelson_chapters": self._list_chapters,
            "read_nelson_chapter": self._read_chapter,
        }.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'."
        try:
            return handler(payload)
        except Exception as exc:  # surfaced to the model, not swallowed
            return f"ERROR running {name}: {type(exc).__name__}: {exc}"

    # -- handlers ---------------------------------------------------------

    def _search(self, payload: dict) -> str:
        query = (payload.get("query") or "").strip()
        if not query:
            return "ERROR: 'query' was empty."
        top_k = _clamp(payload.get("top_k"), 1, 20, 8)
        chapter = payload.get("chapter_number") or None

        result = self.retriever.search(query, top_k=top_k, chapter_number=chapter)
        self._record(result.hits)

        header = f"Search: {query!r}"
        if chapter:
            header += f" (restricted to chapter {chapter})"
        header += f" — {len(result.hits)} passage(s)\n"
        return header + result.to_context_block()

    def _list_chapters(self, payload: dict) -> str:
        contains = payload.get("contains") or None
        limit = _clamp(payload.get("limit"), 1, 100, 40)
        rows = self.retriever.reader.list_chapters(contains, limit)
        if not rows:
            scope = f" matching {contains!r}" if contains else ""
            return f"No chapters{scope} in the index."
        lines = [f"{len(rows)} chapter(s):"]
        for row in rows:
            number = row["chapter_number"] or "?"
            lines.append(f"  Ch. {number}: {row['chapter_title']} (p. {row['page']})")
        return "\n".join(lines)

    def _read_chapter(self, payload: dict) -> str:
        chapter = str(payload.get("chapter_number") or "").strip()
        if not chapter:
            return "ERROR: 'chapter_number' was empty."
        limit = _clamp(payload.get("max_passages"), 1, 25, 12)

        outline = self.retriever.reader.chapter_outline(chapter)
        hits = self.retriever.reader.chapter_chunks(chapter, limit)
        if not hits:
            return (
                f"No chapter numbered {chapter} in the index. "
                "Call list_nelson_chapters to see what is available."
            )
        self._record(hits)

        parts = [f"Chapter {chapter} — {hits[0].chapter_title or '(untitled)'}"]
        if outline:
            parts.append("Sections: " + " | ".join(outline[:40]))
        parts.append(f"Showing {len(hits)} consecutive passage(s):\n")
        parts.extend(hit.as_context(i) for i, hit in enumerate(hits, start=1))
        return "\n".join(parts)

    # -- bookkeeping ------------------------------------------------------

    def _record(self, hits: list[Hit]) -> None:
        seen = {h.chunk_id for h in self.retrieved}
        for hit in hits:
            if hit.chunk_id not in seen:
                self.retrieved.append(hit)
                seen.add(hit.chunk_id)
