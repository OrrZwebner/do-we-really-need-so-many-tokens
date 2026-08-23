"""Command line interface.

    python -m nelson_rag.cli ingest            build the index from data/source
    python -m nelson_rag.cli info              what is in the index
    python -m nelson_rag.cli search "query"    raw retrieval, no model call
    python -m nelson_rag.cli ask "question"    one question, cited answer
    python -m nelson_rag.cli chat              interactive session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings, default_settings

# ANSI styling, disabled when output is piped.
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


DIM, BOLD, CYAN, YELLOW, RED, GREEN = "2", "1", "36", "33", "31", "32"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_ingest(args, settings: Settings) -> int:
    from .indexing.build import build_index

    report = build_index(
        settings,
        source_dir=Path(args.source) if args.source else None,
        progress=lambda msg: print(_c(DIM, msg)),
    )
    print()
    print(_c(GREEN, report.summary()))
    print(_c(DIM, f"Index written to {settings.index_dir}"))
    return 0


def cmd_info(args, settings: Settings) -> int:
    from .indexing.store import IndexReader

    reader = IndexReader(settings.index_dir)
    print(_c(BOLD, "Index"), _c(DIM, str(settings.index_dir)))
    print(f"  chunks       {reader.chunk_count():,}")
    for key in (
        "book_title", "book_edition", "embedder", "embedding_backend",
        "dim", "chunk_chars", "built_at",
    ):
        if key in reader.meta:
            print(f"  {key:<12} {reader.meta[key]}")
    files = reader.meta.get("source_files") or []
    if files:
        print(f"  sources      {', '.join(files)}")
    chapters = reader.list_chapters(limit=10)
    if chapters:
        print(_c(BOLD, "\nFirst chapters"))
        for row in chapters:
            print(f"  Ch. {row['chapter_number'] or '?':<6} {row['chapter_title']}")
    reader.close()
    return 0


def cmd_search(args, settings: Settings) -> int:
    from .session import open_session

    session = open_session(settings)
    result = session.retriever.search(args.query, top_k=args.top_k)
    if not result.hits:
        print(_c(YELLOW, "No passages matched."))
        return 1
    for i, hit in enumerate(result.hits, start=1):
        ranks = f"dense #{hit.dense_rank or '-'} / bm25 #{hit.lexical_rank or '-'}"
        print(_c(CYAN, f"[{i}] {hit.citation}"))
        print(_c(DIM, f"    score {hit.score:.4f}  ({ranks})"))
        body = hit.text if args.full else _truncate(hit.text, 400)
        print("    " + body.replace("\n", "\n    "))
        print()
    session.close()
    return 0


def cmd_ask(args, settings: Settings) -> int:
    from .session import open_session

    session = open_session(settings)
    agent = session.agent()
    answer = _run_question(agent, args.query, show_thinking=args.thinking)
    _print_footer(answer, show_sources=not args.no_sources)
    session.close()
    return 0 if answer.text else 1


def cmd_chat(args, settings: Settings) -> int:
    from .session import open_session

    session = open_session(settings)
    agent = session.agent()
    meta = session.reader.meta
    print(_c(BOLD, f"{meta.get('book_title', 'Nelson')} — clinical reference"))
    print(_c(DIM, f"{session.reader.chunk_count():,} passages indexed. "
                  "Type your question. /reset clears context, /exit quits."))
    while True:
        try:
            question = input(_c(CYAN, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in {"/exit", "/quit", "exit", "quit"}:
            break
        if question == "/reset":
            agent.reset()
            print(_c(DIM, "Context cleared."))
            continue
        try:
            answer = _run_question(agent, question, show_thinking=args.thinking)
            _print_footer(answer, show_sources=not args.no_sources)
        except KeyboardInterrupt:
            print(_c(YELLOW, "\n(interrupted)"))
        except Exception as exc:
            print(_c(RED, f"\nError: {type(exc).__name__}: {exc}"))
    session.close()
    return 0


# --------------------------------------------------------------------------
# shared output helpers
# --------------------------------------------------------------------------


def _run_question(agent, question: str, show_thinking: bool):
    state = {"in_thinking": False}

    def on_event(kind: str, text: str) -> None:
        if kind == "tool":
            print(_c(DIM, f"\n  · {text}"), flush=True)
        elif kind == "warn":
            print(_c(YELLOW, f"\n  ! {text}"), flush=True)
        elif kind == "thinking" and show_thinking:
            if not state["in_thinking"]:
                print(_c(DIM, "\n  [thinking] "), end="", flush=True)
                state["in_thinking"] = True
            print(_c(DIM, text), end="", flush=True)
        elif kind == "text":
            if state["in_thinking"]:
                print("\n", flush=True)
                state["in_thinking"] = False
            print(text, end="", flush=True)

    print()
    answer = agent.ask(question, on_event=on_event)
    print()
    return answer


def _print_footer(answer, show_sources: bool) -> None:
    if answer.refusal:
        print(_c(RED, f"\nRequest declined: {answer.refusal}"))
        return
    if show_sources and answer.hits:
        print(_c(DIM, "\n" + answer.sources_block()))
    warning = answer.grounding.warning_text()
    if warning:
        print(_c(YELLOW, "\nCheck before acting:"))
        print(_c(YELLOW, warning))
    usage = answer.usage
    if usage:
        cached = usage.get("cache_read_input_tokens", 0)
        print(
            _c(
                DIM,
                f"\n[{usage.get('input_tokens', 0):,} in / "
                f"{usage.get('output_tokens', 0):,} out / {cached:,} cached]",
            )
        )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nelson-rag",
        description="Cited clinical Q&A over an indexed pediatrics textbook.",
    )
    parser.add_argument("--index-dir", help="Override the index directory.")
    parser.add_argument("--model", help="Override the answering model.")
    parser.add_argument(
        "--effort", choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort. Higher is slower and more thorough.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build the index from source files.")
    p_ingest.add_argument("--source", help="Directory of source files.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_info = sub.add_parser("info", help="Show what is in the index.")
    p_info.set_defaults(func=cmd_info)

    p_search = sub.add_parser("search", help="Raw retrieval, no model call.")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=8)
    p_search.add_argument("--full", action="store_true", help="Print whole passages.")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="Ask one question.")
    p_ask.add_argument("query")
    p_ask.add_argument("--thinking", action="store_true", help="Stream reasoning.")
    p_ask.add_argument("--no-sources", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive session.")
    p_chat.add_argument("--thinking", action="store_true")
    p_chat.add_argument("--no-sources", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {}
    if args.index_dir:
        overrides["index_dir"] = Path(args.index_dir).expanduser().resolve()
    if args.model:
        overrides["answer_model"] = args.model
    if args.effort:
        overrides["effort"] = args.effort
    settings = default_settings(**overrides)

    try:
        return args.func(args, settings)
    except FileNotFoundError as exc:
        print(_c(RED, str(exc)), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
