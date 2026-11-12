"""A scripted pydantic-ai model for the LLM task's tests.

``Scripted`` replies from a list, or from a function of the prompt's text, and
records every request it gets, so tests can see exactly what the model was sent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

USAGE = RequestUsage(input_tokens=3, output_tokens=2)


def lookup(query: str, id: str = "c1", name: str = "genera_lookup") -> ModelResponse:
    """A reply that calls a lookup tool."""
    return ModelResponse(
        parts=[ToolCallPart(name, {"query": query}, tool_call_id=id)], usage=USAGE
    )


def answer(text: str) -> ModelResponse:
    """A plain text reply."""
    return ModelResponse(parts=[TextPart(text)], usage=USAGE)


def submit(id: str = "s1", **fields: Any) -> ModelResponse:
    """A reply that gives the structured answer, one field per output node."""
    return ModelResponse(
        parts=[ToolCallPart("LLMOutputs", fields, tool_call_id=id)], usage=USAGE
    )


def prompt_parts(messages: list) -> list:
    """The content of the user prompt that opened the conversation."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return list(part.content)
    raise AssertionError("no user prompt")


def prompt_text(messages: list) -> str:
    """The last text block of the opening prompt (the filled-in template)."""
    return [c for c in prompt_parts(messages) if isinstance(c, str)][-1]


class Scripted(FunctionModel):
    """Replies in turn from ``replies``, with ``reply(prompt_text)``, or with
    ``respond(messages)`` given the whole conversation so far.

    A reply that is an exception is raised. ``delay`` makes each request take
    that long, so concurrent requests overlap; ``most_at_once`` records how many
    were ever in flight together.
    """

    def __init__(
        self,
        replies: list | None = None,
        reply: Callable[[str], Any] | None = None,
        respond: Callable[[list], Any] | None = None,
        supports_tools: bool = True,
        delay: float = 0.0,
    ):
        self.replies = list(replies or [])
        self.reply = reply
        self.respond = respond
        self.requests: list[list] = []
        self.infos: list[AgentInfo] = []
        self.delay = delay
        self.in_flight = 0
        self.most_at_once = 0
        super().__init__(
            self._respond,
            model_name="scripted",
            profile={"supports_tools": supports_tools},
        )

    async def _respond(self, messages: list, info: AgentInfo) -> ModelResponse:
        self.requests.append(list(messages))
        self.infos.append(info)
        self.in_flight += 1
        self.most_at_once = max(self.most_at_once, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.respond:
                value = self.respond(messages)
            elif self.reply:
                value = self.reply(prompt_text(messages))
            else:
                value = self.replies.pop(0)
        finally:
            self.in_flight -= 1
        if isinstance(value, Exception):
            raise value
        if isinstance(value, str):
            return answer(value)
        if isinstance(value, dict):
            return submit(**value)
        return value

    @property
    def function_tools(self) -> list[list[str]]:
        return [[tool.name for tool in info.function_tools] for info in self.infos]

    @property
    def output_tools(self) -> list[list[str]]:
        return [[tool.name for tool in info.output_tools] for info in self.infos]


def images(messages: list) -> list[BinaryContent]:
    return [c for c in prompt_parts(messages) if isinstance(c, BinaryContent)]
