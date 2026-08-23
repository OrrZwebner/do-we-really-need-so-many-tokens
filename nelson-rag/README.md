# Nelson RAG — a cited clinical reference assistant

Ask a pediatrics question in plain language (English or Hebrew); get an answer
built **only** from passages retrieved out of your own copy of *Nelson Textbook
of Pediatrics*, with a chapter/section/page citation on every clinical claim.

Built for a practising pediatrician, which drives most of the design decisions
below: it does not hedge, it does not explain what a fever is, and it treats a
fabricated dose as the failure mode worth engineering against.

```
> Two-year-old, 12 kg, acute otitis media, no penicillin allergy. Dose?

  · search_nelson(query='acute otitis media first-line antibiotic dose')
  · search_nelson(query='amoxicillin high dose otitis media children under 2')

High-dose amoxicillin, 90 mg/kg/day divided every 12 hours for 10 days [1].
For a 12 kg child that is 1,080 mg/day → 540 mg every 12 hours.
Use amoxicillin-clavulanate instead if she has had a beta-lactam in the last
30 days or has concurrent purulent conjunctivitis [1].

Sources:
  [1] Nelson Textbook of Pediatrics, 22nd ed., Ch. 220 — Acute Otitis Media,
      TREATMENT, p. 3421

Check before acting:
- These figures are not verbatim in the retrieved passages: 1,080 mg, 540 mg.
  Expected for a dose computed from a weight — otherwise check the source.
```

---

## You supply the book

This repository contains **no** textbook content — Nelson is copyrighted. It
indexes a copy you already have a licence to (an Elsevier/ClinicalKey eBook
export, an institutional PDF, an EPUB).

Drop the file into `data/source/`. Supported: `.pdf`, `.epub`, `.html`, `.txt`,
`.md`. A scanned PDF with no text layer needs OCR first
(`ocrmypdf in.pdf out.pdf`); the ingester will tell you if that's the case.

Everything under `data/` is gitignored so the book never lands in version
control.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # set ANTHROPIC_API_KEY and your edition
mkdir -p data/source && cp ~/nelson.pdf data/source/

python -m nelson_rag.cli ingest      # one-off; ~10-25 min for a full textbook
python -m nelson_rag.cli chat
```

Set `NELSON_BOOK_EDITION` in `.env`. It goes into every citation, and the model
uses it to decide when to warn that current local protocol supersedes the book.

### Commands

| Command | What it does |
|---|---|
| `ingest` | Parse, chunk, embed, and index the source files |
| `info` | Index size, edition, embedder, first chapters |
| `search "query"` | Raw retrieval with scores — **no model call, no API cost** |
| `ask "question"` | One cited answer |
| `chat` | Interactive session with conversation memory |

Useful flags: `--thinking` streams the model's reasoning, `--effort max` for
hard questions, `--full` shows whole passages in `search`.

`streamlit run app.py` gives the same thing in a browser, with retrieved
passages one click from each answer.

---

## How it works

```
 source file
     │  loaders.py      PDF/EPUB/HTML → pages; fix ligatures, rejoin hyphenated
     │                  line-breaks, drop page furniture
     │  structure.py    recover PART / Chapter N / SECTION from the publisher
     │                  ToC and from Nelson's very regular heading style
     │  chunker.py      ~2400-char chunks that never cross a section boundary,
     │                  with overlap and a context header
     ▼
   index/    index.sqlite (metadata + FTS5 BM25) · vectors.npy (float32)
     │
     │  hybrid.py       dense top-30 ∪ BM25 top-30 → Reciprocal Rank Fusion
     │  rerank.py       reorder by IDF-weighted query-term coverage
     ▼
   agent/    Claude Opus 5 + 3 tools, manual loop, adaptive thinking
     │       search_nelson · list_nelson_chapters · read_nelson_chapter
     ▼
   verify.py  re-check the answer against what was actually retrieved
```

### Why hybrid retrieval

Dense embeddings alone lose exactly the tokens that matter clinically. "Give
amoxicillin" and "give ampicillin" are near-neighbours in embedding space and
very different at the bedside. BM25 pins the exact string — drug names,
organisms, `G6PD`, `22q11.2`. Dense catches the paraphrase BM25 misses ("baby
won't stop crying" → the colic chapter). RRF fuses the two rankings without
needing their scores to be comparable, which they aren't.

### Why context headers

A chunk holding the amoxicillin dose contains the word *amoxicillin* and not
the words *acute otitis media* — those are up in the chapter heading. So each
chunk is indexed with a prepended `Chapter 220 Acute Otitis Media > TREATMENT`
header. The header is **not** part of the displayed text, so quotations stay
verbatim. On the test corpus this alone moves the correct passage from rank 2
to rank 1 for `"amoxicillin dose acute otitis media"`.

### Why a manual agent loop

The SDK's tool runner would be less code, but this agent needs to record every
passage handed to the model — for the source list, for the grounding check, and
to tell "answered after searching" from "answered from memory".

---

## The safety design

A medical RAG system fails in a specific way: it produces a fluent, confident,
correctly-formatted answer containing a number that is not in the source. Four
things push against that.

**1. Grounded-only answering.** The system prompt makes retrieval mandatory
before any clinical claim, and requires the model to distinguish *"the
retrieved passages don't cover this"* from *"Nelson doesn't cover this"*.

**2. Citations on every claim.** Inline `[n]` markers plus a source list with
chapter, section, and page. The chunk pipeline exists mainly so those citations
are precise enough to actually look up.

**3. Automatic grounding check** (`agent/verify.py`). After every answer, the
system extracts every dose-shaped number, every `Ch. N`, and every `p. N` from
the answer and checks each against the passages that were really retrieved.
Anything unmatched is surfaced to the clinician. It reports, it does not block —
a dose computed from a weight is legitimately absent from the source, and the
warning says so. What it reliably catches is the dangerous case: a figure that
appears in the answer and nowhere in the book.

**4. Dose rules in the prompt.** Doses must be quoted with route, frequency,
and maximum; never extrapolated from an adult dose, a related drug, or a
different age band; patient-specific arithmetic must be shown.

The model is also told to flag guideline-sensitive topics — immunization
schedules, empiric antibiotics, resuscitation, sepsis bundles — where a
textbook edition lags local protocol.

---

## Configuration

Every knob is an environment variable (see `.env.example`). The ones worth
knowing:

| Variable | Default | Notes |
|---|---|---|
| `NELSON_BOOK_EDITION` | *unspecified* | Set it. Appears in every citation. |
| `NELSON_EMBEDDING_BACKEND` | `local` | `local` (offline, free) · `voyage` (hosted, better) · `hash` (tests only) |
| `NELSON_LOCAL_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Any sentence-transformers model |
| `NELSON_RERANKER` | `coverage` | `cross-encoder` is better if you have the model |
| `NELSON_FUSED_TOP_K` | `8` | Passages per search |
| `NELSON_EFFORT` | `high` | `max` for hard differentials, `low` for lookups |

**Local embeddings are the default on purpose**: the textbook is copyrighted and
the queries are clinical, so nothing is sent anywhere except the passages the
model needs to answer. The index records which embedder built it and refuses to
open under a mismatched one — querying a BGE index with different vectors would
return confident nonsense.

Prompt caching is on: the system prompt and tool definitions are a stable
cached prefix, so a multi-turn conversation re-reads them at ~10% cost.
`cache_read_input_tokens` is printed after each answer.

---

## Tests

```bash
python -m pytest tests/ -q      # 74 tests, ~0.2s, no API key, no downloads
```

They run against a small synthetic corpus in `tests/fixtures/` written in
textbook style — no copyrighted content — using the dependency-free `hash`
embedding backend and a scripted fake Anthropic client. Covered: text
normalisation, chapter/section recovery, chunk boundaries and overlap,
FTS injection-safety on clinical punctuation (`fever >38.5°C`, `22q11.2`,
`amoxicillin/clavulanate`), rank fusion, reranking, tool dispatch, multi-round
agent control flow, refusal handling, dose and citation grounding, and
prompt-cache prefix stability.

---

## Limitations

- **Only as current as your edition.** The assistant flags guideline-sensitive
  answers, but it cannot know your hospital's protocol or local resistance
  patterns.
- **Tables and figures are lossy.** PDF extraction flattens tables; a dosing
  table may arrive as run-together text. Growth charts and images are not
  indexed at all. Treat any table-derived number as needing a look at the page.
- **Retrieval can miss.** If the answer isn't in the top-k, the model is
  instructed to say so rather than improvise — but check the sources on
  anything that matters.
- **Not a decision-support device.** It is a faster way to look something up in
  a book, and it is not regulated, validated, or a substitute for clinical
  judgement.
