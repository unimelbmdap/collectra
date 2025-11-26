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

import llmloader, re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

load_dotenv()


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

    @property
    def input_nums(self) -> int:
        if hasattr(self, "input") and isinstance(self.input, (list, tuple)):
            return len(self.input)
        return 1

    def __init__(self, name: str, model: str, **kwargs):             
        super().__init__(name, **kwargs)
        self.template: str = kwargs.get("template", "")
        self.preamble: str = kwargs.get("preamble", "") 
        self.temperature = kwargs.get("temperature", 0.8)
        self.max_tokens = kwargs.get("max_tokens", None)
        self.llm = llmloader.load(
            model, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens
        )
        self.chain = self.llm | StrOutputParser()

        init_messages = (
            SystemMessage(content=kwargs.get("system", ""))
            if kwargs.get("system", "")
            else SystemMessage(content="You are a helpful assistant.")
        )
        self.messages: list[SystemMessage | HumanMessage] = [init_messages]

    def image_content(self, image: Image):
        """Create image content dictionary for multimodal LLM input.

        Converts an Image instance into the format required by multimodal
        language models, including base64 encoding and MIME type information.

        Args:
            image (Image): The image to convert for LLM processing.

        Returns:
            dict: Dictionary containing image data formatted for LLM input with
                 'type', 'source_type', 'data', and 'mime_type' fields.
        """
        return {
            "type": "image",
            "source_type": "base64",
            "data": image.get_encoding(),
            "mime_type": image.mime(),
        }

    def _add_content(self, value: Image | Text) -> dict:
        if isinstance(value, Image):
            return self.image_content(value)
        return self._add_text(value())

    def _add_text(self, text: str) -> dict:
        return {"type": "text", "text": text}

    def run(self, *args: Text | Image) -> Text:
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
        prompt = f"{self.preamble}\n\n{self.template}".strip()        
        pattern = r"\{(.*?)\}"
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
                if item == key and arg is not None:
                    messages.append(self._add_content(arg))
                    replaced = True
                    break
            if not replaced:
                messages.append(self._add_text(f"No content provided for {item}. Ignore this part."))
            prompt = prompt[end:].strip()
        if prompt:
            messages.append(self._add_text(prompt.strip()))        
        self.messages.append(HumanMessage(content=messages))        
        response = self.chain.invoke(self.messages)
        name = (
            f"{self.get_name()}_output"
            if not hasattr(self, "output")
            else self.output[0] if isinstance(self.output, list) else self.output
        )
        output = Text(name=name, data=response)
        return output

