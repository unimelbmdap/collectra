"""Large Language Model (LLM) tasks for Collectra workflows, built on pydantic-ai.

A task fills its prompt template with the texts and images it is given and
asks a model for the answer: free text for one output node, or one field per
node when there are several. It can call lookups (pipeline resources such as
``EntityLookup``) before it answers.

The model is named as ``provider:model`` (``anthropic:claude-sonnet-5-5``,
``openai:gpt-5``, ``ollama:qwen2.5vl:7b``...). A bare name such as
``claude-sonnet-5-5`` still works for the main hosted providers.

Classes:
    LLM: Task for Large Language Model inference operations
"""

__all__ = ["LLM", "model_id"]

import asyncio
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model, field_validator

from collectra.tasks.text_output import TextOutputTask
from collectra.types.images import Image
from collectra.types.texts import Text

from ..logger import get_logger

# pydantic-ai prints a banner the first time an agent runs; keep output clean.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

load_dotenv()

logger = get_logger(__name__)

DEFAULT_SYSTEM = "You are a helpful assistant."
# How often the model may be asked to correct an answer that does not fit.
RETRIES = 3

# Bare model names that still work, by the provider that serves them.
PROVIDERS = (
    (("claude",), "anthropic"),
    (("gpt", "o1", "o3", "o4", "chatgpt"), "openai"),
    (("gemini", "gemma"), "google"),
    (
        (
            "mistral",
            "pixtral",
            "codestral",
            "ministral",
            "magistral",
            "devstral",
            "open-mistral",
        ),
        "mistral",
    ),
    (("grok",), "xai"),
)


class LLMSetupError(RuntimeError):
    """The task cannot run as configured; raised, not logged page by page."""


class StructuredOutputUnsupported(LLMSetupError):
    """The model cannot produce structured output (it has no tool calling)."""


class ToolCallingUnsupported(LLMSetupError):
    """The model cannot call tools, which a task with ``tools`` needs."""


def model_id(model: str) -> str:
    """``model`` as ``provider:model``, adding the provider to a bare name."""
    if ":" in model:
        return model
    for prefixes, provider in PROVIDERS:
        if model.startswith(prefixes):
            return f"{provider}:{model}"
    raise LLMSetupError(
        f"Cannot tell which provider serves model {model!r}. Give it as "
        "provider:model, for example anthropic:claude-sonnet-5-5, openai:gpt-5, "
        "or ollama:qwen2.5vl:7b for a local model."
    )


def _as_text(cls, value):
    """A field the model left out or filled with a non-string is still a string."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


# Every model call runs on one background event loop, whichever thread asks:
# pydantic-ai's HTTP clients are shared, and must stay on the loop they began on.
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _run(coroutine):
    """Run ``coroutine`` on the shared event loop and wait for its result."""
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever, name="collectra-llm", daemon=True
            ).start()
    return asyncio.run_coroutine_threadsafe(coroutine, _loop).result()


@dataclass
class _Lookups:
    """The lookups made in one conversation, to hold them to ``max_tool_calls``."""

    made: int = 0


def _limited_lookups(task: "LLM", tools: list):
    """The task's lookups as tools, refused past the limit and never raising."""
    from pydantic_ai import ModelRetry
    from pydantic_ai.toolsets import FunctionToolset, WrapperToolset

    class LimitedLookups(WrapperToolset):
        async def call_tool(self, name, tool_args, ctx, tool):
            lookups: _Lookups = ctx.deps
            if lookups.made >= task.max_tool_calls:
                return (
                    f"Lookup limit reached ({task.max_tool_calls}). "
                    "Give your final answer now, without further lookups."
                )
            lookups.made += 1
            logger.info("Task '%s' called %s(%s)", task.name, name, tool_args)
            try:
                return str(await super().call_tool(name, tool_args, ctx, tool))
            except ModelRetry:
                raise
            except Exception as error:
                return f"Lookup failed: {error}"

    return LimitedLookups(FunctionToolset(tools))


class LLM(TextOutputTask):
    """Task for Large Language Model inference operations.

    Attributes:
        name (str): Name identifier for the LLM task.
        model (str): The model, as ``provider:model`` or a bare hosted name.
        template (str | Path): Prompt template string or path to template file.
            ``prompt`` is accepted as an alias, matching other tasks' option name.
        preamble (str): Text put before the template.
        system (str): The system prompt.
        temperature (float | None): Sampling temperature for generation. Leave
            unset (None) to use the model's own default.
        max_tokens (int): Maximum number of tokens to generate.
        cache (bool): Use prompt caching (default True). The fixed text before
            the first image is cached too, so it is reused across pages.
        batch_size (int): Number of prompts sent to the model concurrently when
            the pipeline prefetches many files at once. 1 disables batching.
        output: One node name, or several. With several, the model answers in
            structured form: one string field per node, named after the node,
            and every node gets a Text (empty when the model found nothing).
            The prompt template is where each field is described.
        tools: Names of pipeline resources (such as ``EntityLookup``) the model
            may call before answering. A task with tools needs a model that
            supports tool calling. With several outputs the answer is given by
            calling one more tool, whose fields are the outputs.
        max_tool_calls (int): The most lookups the model may make per prompt.
    """

    def __init__(self, name: str, model: str, **kwargs):
        super().__init__(name, **kwargs)
        if kwargs.get("template") and kwargs.get("prompt"):
            raise ValueError(
                f"LLM task {name!r} sets both template and prompt; use only one"
            )
        self.template: str = kwargs.get("template") or kwargs.get("prompt", "")
        self.preamble: str = kwargs.get("preamble", "")
        # None leaves the model at its own default: some models reject the setting.
        self.temperature = kwargs.get("temperature")
        self.max_tokens = kwargs.get("max_tokens", None)
        self.cache = bool(kwargs.get("cache", True))
        self.model = model
        self.batch_size = int(self.batch_size)  # the Task base requires >= 1
        self._prefetched: dict[str, list] = {}
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError(f"LLM task {name!r} lists the same output more than once")
        tools = kwargs.get("tools")
        self.tools = (
            [tools] if isinstance(tools, str) else [str(t) for t in tools or []]
        )
        self.max_tool_calls = int(kwargs.get("max_tool_calls", 5))
        if self.max_tool_calls < 1:
            raise ValueError(f"LLM task {name!r} needs max_tool_calls >= 1")

        self.system: str = kwargs.get("system", "") or DEFAULT_SYSTEM
        # The system prompt, then (once a prompt is built) the user's content.
        self.messages: list = [self.system]

    @cached_property
    def llm(self):
        """The pydantic-ai model, loaded when it is first needed."""
        try:
            from pydantic_ai.models import infer_model
        except ModuleNotFoundError as error:
            if error.name != "pydantic_ai":
                raise
            raise ImportError(
                'LLM execution requires optional dependencies. Install with: pip install "collectra[ml]"'
            ) from error
        return infer_model(model_id(self.model))

    @property
    def output_names(self) -> list[str]:
        """The output node names, whether given as one name or a list."""
        output = getattr(self, "output", None)
        if output is None:
            return []
        return [output] if isinstance(output, str) else list(output)

    @property
    def is_structured(self) -> bool:
        """Several output nodes mean the model must answer field by field."""
        return len(self.output_names) > 1

    @cached_property
    def output_model(self) -> type[BaseModel]:
        """One optional string field per output node, named after the node.

        Fields are positional (``field_0``...) with the node name as alias, so
        any node name works, even ones pydantic would reject as an attribute.
        """
        fields = {
            f"field_{index}": (str, Field(default="", alias=name))
            for index, name in enumerate(self.output_names)
        }
        return create_model(
            "LLMOutputs",
            __doc__=(
                "Fill in every field from the input. Leave a field as an empty "
                "string when the input does not contain it."
            ),
            __validators__={"as_text": field_validator("*", mode="before")(_as_text)},
            **fields,
        )

    def model_settings(self) -> dict[str, Any]:
        """The settings sent with every request: only those that were set."""
        settings: dict[str, Any] = {}
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        settings["cache"] = self.cache
        return settings

    @cached_property
    def agent(self):
        """The pydantic-ai agent that asks the model, with the task's tools."""
        from pydantic_ai import Agent, ToolOutput

        model = self.llm
        supports_tools = model.profile.get("supports_tools", True)
        if self.tools and not supports_tools:
            raise ToolCallingUnsupported(
                f"LLM task {self.name!r} uses tools, but model {self.model!r} "
                "does not support tool calling. Use a model that does, or remove "
                "the tools."
            )
        if self.is_structured and not supports_tools:
            raise StructuredOutputUnsupported(
                f"LLM task {self.name!r} has {len(self.output_names)} outputs, which "
                f"needs structured output, but model {self.model!r} does not support "
                "tool calling. Use a model that does, or give the task one output."
            )
        toolsets = []
        if self.tools:
            toolsets.append(_limited_lookups(self, list(self._tool_objects().values())))
        # With several outputs the answer itself is a tool call: the model is
        # free to look things up first and submits when it is done.
        output_type = (
            ToolOutput(self.output_model, name=self.output_model.__name__)
            if self.is_structured
            else str
        )
        return Agent(
            model,
            output_type=output_type,
            instructions=self.system,
            toolsets=toolsets,
            deps_type=_Lookups,
            model_settings=self.model_settings(),
            retries=RETRIES,
            # An answer ends the conversation, even with lookups in the same reply.
            end_strategy="early",
        )

    def validate_resources(self, resources: dict[str, Any]) -> None:
        """Fail early, when the pipeline connects, if a named tool does not exist."""
        for tool in self.tools:
            if tool not in resources:
                raise ValueError(
                    f"Task {self.name!r} uses unknown tool {tool!r}. "
                    f"Available tools: {', '.join(sorted(resources)) or 'none'}"
                )

    def _tool_objects(self) -> dict[str, Any]:
        """The task's tools by name, taken from the pipeline's resources."""
        if self.pipeline is None:
            raise LLMSetupError(
                f"LLM task {self.name!r} uses tools but is not part of a pipeline"
            )
        tools = [self.pipeline.resource(name).as_tool() for name in self.tools]
        return {tool.name: tool for tool in tools}

    # --- the prompt ---------------------------------------------------------------

    def get_pattern_matches(self) -> tuple:
        prompt = f"{self.preamble}\n\n{self.template}".strip()
        if not prompt:
            raise ValueError(
                f"LLM task {self.name!r} has no prompt; set template: (or prompt:) "
                "in its pipeline.yaml entry"
            )
        pattern = r"\{(.*?)\}"
        return prompt, pattern

    def image_content(self, image: Image) -> dict:
        """The image as a content block, in a format models accept."""
        return {"type": "image", "data": image.load(), "mime_type": image.mime()}

    def _add_content(self, value: Image | Text) -> dict:
        if isinstance(value, Image):
            return self.image_content(value)
        return self._add_text(value())

    def _add_text(self, text: str) -> dict:
        text = (
            "''" if len(text.strip()) == 0 else text.strip()
        )  # Avoid empty string issues in some LLMs
        return {"type": "text", "text": text}

    def pre_run(self) -> tuple:
        self.messages = self.messages[:1]  # Reset to the system prompt
        prompt, pattern = self.get_pattern_matches()
        return prompt, pattern

    def check_inputs(self, *args: Text | Image) -> None:
        """Validate that the messages are not longer than the expected number of inputs."""
        residual_inputs = len(args)
        if len(self.messages) != 2:
            raise ValueError("Expected one system message and one human message.")
        for content in self.messages[1]:
            if isinstance(content, dict) and (
                content.get("type") == "image" or content.get("type") == "text"
            ):
                residual_inputs -= 1
        if residual_inputs > 0:
            raise ValueError("Number of inputs does not match the expected count")

    def replace_inputs(
        self, pattern: str, prompt: str, *args: Text | Image, **kwargs
    ) -> list[str | dict]:
        messages: list[str | dict] = list()
        while re.search(pattern, prompt):
            match = next(re.finditer(pattern, prompt))
            start, end = match.span()
            item = match[1].strip()
            replaced = False
            if prompt[:start]:
                messages.append(self._add_text(prompt[:start]))
            for arg in args:
                key = arg.name
                if key == item:
                    messages.append(self._add_content(arg))
                    replaced = True
                    break
            if item == "entities" and kwargs.get("entities", None):
                replaced = True
                messages.append(self._add_text(kwargs["entities"]))
            if not replaced:
                messages.append(
                    self._add_text(f"No content provided for {item}. Ignore this part.")
                )
            prompt = prompt[end:].strip()

        if prompt:
            messages.append(self._add_text(prompt))

        return messages

    def _build_messages(self, args: tuple) -> list[str | dict]:
        """Fill the prompt template for ``args``, leaving ``self.messages`` ready to send."""
        prompt, pattern = self.pre_run()
        content = self.replace_inputs(pattern, prompt, *args)
        self.messages.append(content)
        self.check_inputs(*args)
        return content

    def _user_content(self, content: list[dict]) -> list:
        """The prompt's blocks as pydantic-ai content, in the same order.

        With caching on, a cache point goes just before the first image, so the
        fixed text of the template ahead of it is reused from page to page.
        """
        from pydantic_ai.messages import BinaryContent, CachePoint

        parts: list = []
        for block in content:
            if block.get("type") == "image":
                if self.cache and parts and all(isinstance(p, str) for p in parts):
                    parts.append(CachePoint())
                parts.append(
                    BinaryContent(data=block["data"], media_type=block["mime_type"])
                )
            else:
                parts.append(block["text"])
        return parts

    @staticmethod
    def _cache_key(content: list[str | dict]) -> str:
        """Identify a prompt by its content (images are hashed, not stored)."""

        def digest(value: bytes) -> str:  # only an image's bytes need this
            return hashlib.sha256(value).hexdigest()

        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=digest).encode()
        ).hexdigest()

    # --- asking the model -------------------------------------------------------------

    async def _ask(self, content: list[dict]):
        """One conversation with the model for ``content``; its run result."""
        from pydantic_ai import UsageLimits
        from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

        lookups = _Lookups()
        # Room for every allowed lookup, then a few turns to give the answer.
        limits = UsageLimits(request_limit=self.max_tool_calls + 3)
        try:
            return await self.agent.run(
                self._user_content(content), deps=lookups, usage_limits=limits
            )
        except UsageLimitExceeded as error:
            raise RuntimeError(
                f"LLM task {self.name!r}: the model did not give a final answer "
                f"after {lookups.made} lookups"
            ) from error
        except UnexpectedModelBehavior as error:
            raise RuntimeError(
                f"LLM task {self.name!r}: the model did not give a final answer "
                f"({error})"
            ) from error

    async def _ask_all(self, contents: list[list[dict]]) -> list:
        """Ask for every prompt, ``batch_size`` at a time; failures are returned."""
        slots = asyncio.Semaphore(self.batch_size)

        async def one(content):
            async with slots:
                return await self._ask(content)

        return await asyncio.gather(*(one(c) for c in contents), return_exceptions=True)

    def prefetch(self, entries: list[list]) -> None:
        """Send the prompts for many entries to the model, ``batch_size`` at a time.

        Results are cached by prompt content and consumed by ``run``, so usage
        is recorded against the file that is actually being processed. A prompt
        that fails here is simply left for ``run`` to retry on its own.
        """
        self._prefetched = {}
        prompts: list[tuple[str, list]] = []
        for entry in entries:
            if not entry:
                continue
            try:
                content = self._build_messages(tuple(entry))
            except Exception as e:
                logger.warning(
                    "LLM task '%s' could not prepare an entry: %s", self.name, e
                )
                continue
            prompts.append((self._cache_key(content), content))
        if not prompts:
            return
        self.agent  # a setup problem is raised now, not logged for every prompt
        results = _run(self._ask_all([content for _, content in prompts]))
        for (key, _), result in zip(prompts, results):
            if isinstance(result, Exception):
                logger.warning(
                    "LLM task '%s' prompt failed, will retry alone: %s",
                    self.name,
                    result,
                )
                continue
            self._prefetched.setdefault(key, []).append(result)

    def clear_prefetch(self) -> None:
        self._prefetched = {}

    def _fields(self, parsed: BaseModel) -> dict[str, str]:
        """The structured answer as ``{node name: text}``, in output order."""
        values = {}
        for index, name in enumerate(self.output_names):
            value = getattr(parsed, f"field_{index}")
            values[name] = "" if value in ('""', "''") else value.strip()
        return values

    def _record_usage(self, usage) -> None:
        """Add one conversation's token usage to the usage file, if one is in use.

        ``input_tokens`` includes the tokens read from and written to the cache.
        """
        if not self.context.usage_file:
            return
        usage_file = self.context.usage_file
        data = {
            f"{self.name}": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
            }
        }
        old_data = {}
        if usage_file.exists():
            with open(usage_file, "r") as f:
                old_data = yaml.safe_load(f) or dict()
        for key, value in old_data.items():
            if key not in data:
                data[key] = value
            else:
                for item, item_value in value.items():
                    data[key][item] = data.get(key, {}).get(item, 0) + item_value
        usage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(usage_file, "w") as f:
            yaml.dump(data, f)

    def invoke(self, response=None) -> str | dict[str, str]:
        """Ask the model (unless ``response`` is already in hand) and read the reply.

        Returns text for a single output, or ``{node name: text}`` for several.
        """
        if response is None:
            self.agent  # a setup problem is raised here, in the caller's thread
            response = _run(self._ask(self.messages[1]))
        self._record_usage(response.usage)
        if self.is_structured:
            return self._fields(response.output)
        output = response.output
        return "" if output in ('""', "''") else output.strip()

    def run(self, *args: Text | Image) -> Text | list[Text] | None:
        """Fill the template with ``args``, ask the model and return its answer.

        A failed call or an unreadable reply raises: the pipeline then records
        the failure and saves no output, so the next run tries again. Returning
        nothing would be saved as "found nothing" and never retried.
        """
        content = self._build_messages(args)

        cached = self._prefetched.get(self._cache_key(content))
        response = self.invoke(cached.pop(0) if cached else None)

        if self.is_structured:
            return [
                self.text_output(value, name=node) for node, value in response.items()
            ]
        return self.text_output(response)
