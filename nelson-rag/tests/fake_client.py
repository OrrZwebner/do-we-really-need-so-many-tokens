"""A stand-in for `anthropic.Anthropic` that replays scripted turns.

Lets the agent loop be tested for real — tool dispatch, multi-round control
flow, usage accounting, refusal handling, prompt-cache stability — with no
network and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Delta:
    type: str
    text: str = ""
    thinking: str = ""


@dataclass
class Event:
    type: str
    delta: Delta | None = None


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class StopDetails:
    type: str = "refusal"
    category: str | None = None
    explanation: str = ""


@dataclass
class FinalMessage:
    content: list
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    stop_details: StopDetails | None = None


class _Stream:
    def __init__(self, turn: FinalMessage):
        self._turn = turn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for block in self._turn.content:
            if block.type == "text":
                yield Event("content_block_delta", Delta("text_delta", text=block.text))
            elif block.type == "thinking":
                yield Event(
                    "content_block_delta",
                    Delta("thinking_delta", thinking=getattr(block, "thinking", "")),
                )

    def get_final_message(self) -> FinalMessage:
        return self._turn


class _Messages:
    def __init__(self, client: "FakeClient", beta: bool):
        self._client = client
        self._beta = beta

    def stream(self, **kwargs):
        if self._beta and self._client.reject_beta:
            raise TypeError("unexpected keyword argument 'fallbacks'")
        # Snapshot the message list: the real API serialises at call time,
        # and the agent keeps mutating its own list afterwards.
        self._client.requests.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        if not self._client.turns:
            raise AssertionError("FakeClient ran out of scripted turns")
        return _Stream(self._client.turns.pop(0))


class _Beta:
    def __init__(self, client: "FakeClient"):
        self.messages = _Messages(client, beta=True)


class FakeClient:
    """Replays ``turns`` in order, recording the request kwargs it received."""

    def __init__(self, turns: list[FinalMessage], reject_beta: bool = False):
        self.turns = list(turns)
        self.reject_beta = reject_beta
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(self, beta=False)
        self.beta = _Beta(self)


def text_turn(text: str, **kw) -> FinalMessage:
    return FinalMessage(content=[TextBlock(text)], stop_reason="end_turn", **kw)


def tool_turn(name: str, payload: dict, tool_id: str = "tu_1", preamble: str = "") -> FinalMessage:
    content: list = [TextBlock(preamble)] if preamble else []
    content.append(ToolUseBlock(tool_id, name, payload))
    return FinalMessage(content=content, stop_reason="tool_use")
