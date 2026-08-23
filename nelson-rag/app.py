"""Streamlit front end.

    streamlit run app.py

A chat window over the same agent the CLI uses. The extra thing a GUI buys
here is that the retrieved passages sit one click away from every answer, so
the clinician can read the textbook's own words rather than trusting the
summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nelson_rag.config import default_settings  # noqa: E402
from nelson_rag.session import open_session  # noqa: E402

st.set_page_config(page_title="Pediatrics reference", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading index and embedding model...")
def get_session():
    return open_session(default_settings())


def main() -> None:
    try:
        session = get_session()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    meta = session.reader.meta
    _sidebar(session, meta)

    st.title(meta.get("book_title", "Pediatrics reference"))
    st.caption(
        f"{session.reader.chunk_count():,} passages indexed from "
        f"{meta.get('book_edition', 'the source text')}. "
        "Every answer is drawn from retrieved passages and cited; "
        "verify doses against the source before acting on them."
    )

    if "agent" not in st.session_state:
        st.session_state.agent = session.agent()
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("answer"):
                _render_answer_details(entry["answer"])

    question = st.chat_input("Ask a clinical question…")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Searching the textbook…", expanded=False)
        placeholder = st.empty()
        buffer: list[str] = []

        def on_event(kind: str, text: str) -> None:
            if kind == "tool":
                status.write(f"🔎 {text}")
            elif kind == "warn":
                status.write(f"⚠️ {text}")
            elif kind == "text":
                buffer.append(text)
                placeholder.markdown("".join(buffer))

        try:
            answer = st.session_state.agent.ask(question, on_event=on_event)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return

        status.update(label=f"Searched · {len(answer.hits)} passages", state="complete")
        if answer.refusal:
            st.error(f"Request declined: {answer.refusal}")
            return
        placeholder.markdown(answer.text)
        _render_answer_details(answer)

    st.session_state.history.append(
        {"role": "assistant", "content": answer.text, "answer": answer}
    )


def _render_answer_details(answer) -> None:
    warning = answer.grounding.warning_text()
    if warning:
        st.warning("**Check before acting:**\n\n" + warning)

    if answer.hits:
        # Unnumbered on purpose: the model's [n] markers are local to each
        # search, so numbering here would imply a mapping that does not hold.
        with st.expander(f"Sources — {len(answer.hits)} passage(s) retrieved"):
            for hit in answer.hits:
                st.markdown(f"**{hit.citation}**")
                st.text(hit.text)
                st.divider()

    if answer.usage:
        st.caption(
            f"{answer.usage.get('input_tokens', 0):,} in · "
            f"{answer.usage.get('output_tokens', 0):,} out · "
            f"{answer.usage.get('cache_read_input_tokens', 0):,} cached"
        )


def _sidebar(session, meta: dict) -> None:
    with st.sidebar:
        st.subheader("Index")
        st.write(f"**Source:** {meta.get('book_edition', 'unknown edition')}")
        st.write(f"**Passages:** {session.reader.chunk_count():,}")
        st.write(f"**Embeddings:** {meta.get('embedder', '?')}")
        st.write(f"**Built:** {meta.get('built_at', '?')}")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.subheader("Browse chapters")
        needle = st.text_input("Filter by title", placeholder="e.g. seizure")
        for row in session.reader.list_chapters(needle or None, limit=25):
            st.caption(f"Ch. {row['chapter_number'] or '?'} — {row['chapter_title']}")


if __name__ == "__main__":
    main()
