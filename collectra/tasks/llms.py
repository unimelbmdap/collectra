"""Large Language Model (LLM) integration tasks for Collectra workflows.

This module provides integration with various Large Language Models through
the LangChain framework. It supports both text and multimodal (text+image)
processing tasks with configurable prompts and model parameters.

The module includes:
    - LLM task class with configurable model backends
    - Support for text and image inputs
    - Template-based prompt formatting
    - Integration with various LLM providers via llmloader

Classes:
    LLM: Task for Large Language Model inference operations
"""

__all__ = ["LLM"]

import os
import re
from pathlib import Path

import llmloader
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

from ..logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class LLM(Task):
    """Task for Large Language Model inference operations.

    Integrates with various LLM providers to perform text generation and
    multimodal processing tasks. Supports configurable prompts, temperature,
    and token limits with template-based input formatting.

    Attributes:
        name (str): Name identifier for the LLM task.
        model (str): Model identifier or path for the LLM.
        template (str | Path): Prompt template string or path to template file.
        temperature (float): Sampling temperature for generation (0.0 to 1.0).
        max_tokens (int): Maximum number of tokens to generate.
        variables (dict): Additional variables for template substitution.
    """

    def __init__(self, name: str, model: str, **kwargs):
        super().__init__(name, **kwargs)
        self.template: str = kwargs.get("template", "")
        self.preamble: str = kwargs.get("preamble", "")
        self.temperature = kwargs.get("temperature", 0.8)
        self.max_tokens = kwargs.get("max_tokens", None)
        self.llm = llmloader.load(
            model, temperature=self.temperature, max_tokens=self.max_tokens
        )

        self.chain = self.llm
        self.parser = StrOutputParser()

        init_messages = (
            SystemMessage(content=kwargs.get("system", ""))
            if kwargs.get("system", "")
            else SystemMessage(content="You are a helpful assistant.")
        )
        self.messages: list[SystemMessage | HumanMessage] = [init_messages]

    def get_pattern_matches(self) -> tuple:
        prompt = f"{self.preamble}\n\n{self.template}".strip()
        pattern = r"\{(.*?)\}"
        return prompt, pattern

    def image_content(self, image: Image):
        return llmloader.LLMWrapper.format(
            self.llm,
            data_type="image",
            data={
                "data": image.get_encoding(),
                "mime_type": image.mime(),
            },
        )

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
        self.messages = self.messages[:1]  # Reset to initial system message
        prompt, pattern = self.get_pattern_matches()
        return prompt, pattern

    def check_inputs(self, *args: Text | Image) -> None:
        """Validate that the messages are not longer than the expected number of inputs."""
        residual_inputs = len(args)
        if len(self.messages) != 2:
            raise ValueError("Expected one system message and one human message.")
        human_message = self.messages[1]
        for content in human_message.content:
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

    def invoke(self) -> str:
        response = self.chain.invoke(self.messages)

        # Use context for usage tracking instead of environment variable
        if self.context.usage_file:
            usage = llmloader.LLMWrapper.get_token_count(response)
            usage_file = self.context.usage_file
            data = {f"{self.name}": usage}
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

        response = self.parser.invoke(response)

        return "" if response in ('""', "''") else response.strip()

    def run(self, *args: Text | Image) -> Text | None:
        """Execute LLM inference on the provided inputs with template-based prompt generation.

        For list inputs, each item is processed individually and results are collected
        in a list. The method supports multimodal inputs (text + images) by creating
        appropriate message structures for the LangChain conversation format.

        Args:
            **kwargs: Input data for LLM processing. Keys should match template
                     placeholders. Values can be text strings, file paths (Path objects),
                     Image objects, or lists of these types for batch processing.

        Side Effects:
            Updates self.output dictionary with generated text responses. Output keys
            that contain the input key as a substring will be populated with LLM results.
            Prints a success message when inference completes.
        """
        try:
            prompt, pattern = self.pre_run()

            messages = self.replace_inputs(pattern, prompt, *args)

            self.messages.append(HumanMessage(content=messages))

            self.check_inputs(*args)

            response = self.invoke()

            name = self.get_output_name()
            output = Text(name=name, data=response)
            return output
        except Exception as e:
            logger.error("LLM task '%s' failed: %s", self.name, e, exc_info=True)
            return None
