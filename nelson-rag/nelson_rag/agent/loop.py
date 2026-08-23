"""The agentic loop: Claude + the Nelson tools, with citations and grounding checks.

A manual loop rather than the SDK tool runner, because this agent needs three
things the runner does not expose: a record of every passage handed to the
model (for the grounding check and the source list), streaming callbacks for
the CLI, and the ability to tell "answered without searching" from "answered
after searching".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..config import Settings
from ..indexing.store import Hit
from ..retrieval.hybrid import HybridRetriever
from .prompts import build_system_prompt
from .tools import TOOLS, ToolRunner
from .verify import GroundingReport, check_answer

# Server-side refusal fallback: if a safety classifier declines a request, the
# API retries on another model rather than handing back an empty turn. Medical
# questions sit near enough to the classifiers that this is worth having on.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"

EventCallback = Callable[[str, str], None]


@dataclass
class Answer:
    text: str
    thinking: str
    hits: list[Hit]
    tool_calls: list[dict]
    grounding: GroundingReport
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    refusal: str | None = None

    def sources_block(self) -> str:
        """Every passage the model saw, deliberately unnumbered.

        The model's own [1], [2] markers are local to each tool result, so
        numbering this list too would invite the reader to line them up when
        they do not correspond. The model's Sources section carries the
        mapping; this is the wider set it had available.
        """
        if not self.hits:
            return ""
        lines = [f"Passages retrieved ({len(self.hits)}):"]
        lines.extend(f"  · {hit.citation}" for hit in self.hits)
        return "\n".join(lines)


class NelsonAgent:
    """Multi-turn clinical Q&A over the indexed textbook."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        client=None,
    ):
        self.settings = settings
        self.retriever = retriever
        self.messages: list[dict] = []
        self._client = client
        # Whether this SDK/account accepts the refusal-fallback beta. Probed
        # once on the first request, then remembered.
        self._fallback_supported: bool | None = None

        meta = retriever.reader.meta
        self.system_prompt = build_system_prompt(
            meta.get("book_title", settings.book_title),
            meta.get("book_edition", settings.book_edition),
        )

    # -- client ------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:  # pragma: no cover - optional at index time
                raise RuntimeError(
                    "Answering needs the Anthropic SDK: pip install anthropic"
                ) from None
            self._client = anthropic.Anthropic()
        return self._client

    def reset(self) -> None:
        """Clear conversation history; the index and settings stay."""
        self.messages = []

    # -- main entry point --------------------------------------------------

    def ask(self, question: str, on_event: EventCallback = lambda kind, text: None) -> Answer:
        runner = ToolRunner(self.retriever)
        # Where this turn's messages begin. If the turn fails or is refused we
        # rewind to here: a trailing user message with no assistant reply would
        # make the *next* question an invalid request.
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": question})

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        usage: dict = {}
        stop_reason: str | None = None
        refusal: str | None = None

        for _round in range(self.settings.max_tool_rounds):
            try:
                response = self._stream_turn(on_event, text_parts, thinking_parts)
            except Exception:
                self.messages = self.messages[:turn_start]
                raise
            stop_reason = response.stop_reason
            _accumulate_usage(usage, response.usage)

            if stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                refusal = getattr(details, "explanation", None) or (
                    "The request was declined by a safety classifier."
                )
                self.messages = self.messages[:turn_start]
                break

            self.messages.append({"role": "assistant", "content": response.content})

            if stop_reason == "pause_turn":
                # A long server-side turn paused; re-send to let it continue.
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            results = []
            for block in tool_uses:
                payload = block.input if isinstance(block.input, dict) else json.loads(block.input)
                on_event("tool", f"{block.name}({_short(payload)})")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": runner.run(block.name, payload),
                    }
                )
            self.messages.append({"role": "user", "content": results})
        else:
            on_event(
                "warn",
                f"Stopped after {self.settings.max_tool_rounds} tool rounds "
                "without a final answer.",
            )

        answer_text = "".join(text_parts).strip()
        searched = any(c["tool"] != "list_nelson_chapters" for c in runner.calls)
        grounding = check_answer(answer_text, runner.retrieved, searched=searched)

        return Answer(
            text=answer_text,
            thinking="".join(thinking_parts).strip(),
            hits=runner.retrieved,
            tool_calls=runner.calls,
            grounding=grounding,
            usage=usage,
            stop_reason=stop_reason,
            refusal=refusal,
        )

    # -- one streamed request ---------------------------------------------

    def _stream_turn(
        self,
        on_event: EventCallback,
        text_parts: list[str],
        thinking_parts: list[str],
    ):
        kwargs = dict(
            model=self.settings.answer_model,
            max_tokens=self.settings.max_tokens,
            # Caching the system block also covers the tool definitions, which
            # render ahead of it and never change between turns.
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.settings.effort},
            tools=TOOLS,
            messages=self.messages,
        )

        for attempt in self._request_variants(kwargs):
            try:
                with attempt() as stream:
                    for event in stream:
                        if event.type != "content_block_delta":
                            continue
                        delta = event.delta
                        if delta.type == "text_delta":
                            text_parts.append(delta.text)
                            on_event("text", delta.text)
                        elif delta.type == "thinking_delta":
                            thinking_parts.append(delta.thinking)
                            on_event("thinking", delta.thinking)
                    return stream.get_final_message()
            except Exception as exc:
                if self._is_unsupported_param(exc) and self._fallback_supported is not False:
                    # This SDK build or account does not accept the beta;
                    # fall through to the plain request and stop trying.
                    self._fallback_supported = False
                    on_event("warn", "Refusal fallback unavailable; continuing without it.")
                    continue
                raise
        raise RuntimeError("no request variant succeeded")  # pragma: no cover

    def _request_variants(self, kwargs: dict) -> Iterable[Callable]:
        """Preferred request shape first, then a plain one if the beta is rejected."""
        if self._fallback_supported is not False:
            def with_fallback():
                self._fallback_supported = True
                return self.client.beta.messages.stream(
                    **kwargs, betas=[REFUSAL_FALLBACK_BETA], fallbacks="default"
                )
            yield with_fallback
        yield lambda: self.client.messages.stream(**kwargs)

    @staticmethod
    def _is_unsupported_param(exc: Exception) -> bool:
        message = str(exc).lower()
        signals = ("fallbacks", "unexpected keyword", "betas", "unsupported", "beta")
        return any(s in message for s in signals)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _accumulate_usage(total: dict, usage) -> None:
    if usage is None:
        return
    for field_name in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, field_name, None)
        if value:
            total[field_name] = total.get(field_name, 0) + int(value)


def _short(payload: dict, limit: int = 80) -> str:
    rendered = ", ".join(
        f"{k}={v!r}" for k, v in payload.items() if v not in (None, "")
    )
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"
