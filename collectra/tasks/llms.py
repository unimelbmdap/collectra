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

from collectra.tasks.base import Task
from collectra.images.base import Image
from collectra.utils import success_msg, processing_msg
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from rich import print
import copy, llmloader

load_dotenv()

@dataclass(kw_only=True)
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

    name: str
    model: str
    template: str | Path    
    temperature: float = 0.8
    max_tokens: int = 250
    variables: dict = field(default_factory=dict)

    def input_type(self) -> type | tuple:
        """Define the expected input types for this LLM task.
        
        Returns:
            type | tuple: Tuple of str and Path types for text and file inputs.
        """
        return str, Path

    def output_type(self) -> type | tuple:
        """Define the expected output types for this LLM task.
        
        Returns:
            type | tuple: String type for generated text outputs.
        """
        return str

    def process_inputs(self, **kwargs) -> tuple:
        """Process and merge input arguments with predefined variables.
        
        Combines the provided keyword arguments with the task's predefined
        variables to create a unified set of template substitution parameters.
        
        Args:
            **kwargs: Input arguments provided to the task.
            
        Returns:
            tuple: Items from the merged dictionary of variables and inputs.
        """
        kwargs.update(self.variables)     
        return kwargs.items() # type: ignore

    def __post_init__(self):
        """Initialize the LLM instance and message templates after object creation.
        
        Loads the specified LLM model using the llmloader library and sets up
        the initial system message for the conversation context.
        """
        self.llm = llmloader.load(Path(self.model).name, temperature=self.temperature, max_tokens=self.max_tokens)           
        self.messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=self.config.get("system", ""))
        ]    
    
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

    def add_text(self, prompt: str, key: str, value: str=""):
        """Create text content dictionary with template substitution.
        
        Processes a prompt template by replacing placeholders with actual values
        and formats it for LLM input.
        
        Args:
            prompt (str): The prompt template string containing placeholders.
            key (str): The key to replace in the template (e.g., for {key} placeholder).
            value (str, optional): The value to substitute for the key. Defaults to "".
            
        Returns:
            dict: Dictionary containing formatted text for LLM input with
                 'type' and 'text' fields.
        """
        return {
            "type": "text",
            "text": prompt.replace(f"{{{key}}}", value).replace("{input}", value).strip()
        }

    def replace_in_template(self, prompt, key, value) -> list[dict | str]:
        """Replace template placeholders with appropriate content based on value type.
        
        Handles different types of input values (images, files, text) and creates
        the appropriate content structure for LLM processing. Supports multimodal
        inputs by detecting images and converting them to the proper format.
        
        Args:
            prompt: The prompt template string to process.
            key: The template key to replace.
            value: The value to substitute, can be Image, Path, or string.
            
        Returns:
            list[dict | str]: List of content dictionaries formatted for LLM input,
                            may include both text and image content.
        """
        content = []                        
        if isinstance(value, Image):                  
            content.append(self.add_text(prompt, key))      
            content.append(self.image_content(value))                                                 
        elif Path(value).is_file():       
            try:                    
                image = Image.build(Path(value))                         
                content.append(self.add_text(prompt, key))      
                content.append(self.image_content(image))                                 
            except Exception as e:                
                value = value.read_text()                    
                content.append(self.add_text(prompt, key, str(value)))     
        else:                
            content.append(self.add_text(prompt, key, str(value)))     
        return content
    
    def run(self, **kwargs):
        """Execute LLM inference on the provided inputs.
        
        Processes all input arguments through the template system, creates
        appropriate message structures for the LLM, and generates responses
        for each input-output mapping defined in the task configuration.
        
        Args:
            **kwargs: Input data for LLM processing. Can include text strings,
                     file paths, Image objects, or other data types supported
                     by the template system.
        
        Side Effects:
            Updates self.output with generated text responses for each configured
            output key that matches the input keys.
        """
        prompt = str(self.template)                       
        items = self.process_inputs(**kwargs)
        parser = StrOutputParser()
        results = dict()
        for key, value in items:
            if not bool(value):
                continue
            content = self.replace_in_template(prompt, key, value)     
            message = self.messages.copy()
            message.append(HumanMessage(content=content))                        
            for output_key in self.output:
                if key in output_key:
                    self.output[output_key] = parser.invoke(self.llm.invoke(message))                                    
        print(success_msg(f"[yellow]Inference Complete.[/yellow]"))            
        

    def save(self, output_path: Path, **kwargs) -> tuple[Path | None, dict, list[Path]]:
        """Save LLM task results by merging generated text with existing output data.
        
        Processes the generated LLM outputs and merges them with existing output
        data structures. The method looks for output keys ending with '_text'
        and updates the corresponding data entries with the generated text.
        
        Args:
            output_path (Path): Directory path where results should be saved.
            **kwargs: Additional arguments including 'output_data' containing
                     existing output data to merge with LLM results.
        
        Returns:
            tuple[Path | None, dict, list[Path]]: A tuple containing:
                - None (no specific output file path for LLM results)
                - dict: Merged result data with LLM-generated text included
                - list[Path]: Empty list (no additional files created)
        """
        output_data = kwargs.get("output_data", dict())
        result_data = dict()
        for key in self.output:
            data_key = key.replace("_text", "")
            if data_key in output_data:
                to_be_updated = copy.deepcopy(output_data[data_key])
                to_be_updated.update({"text": str(self.output[key])})
                result_data[data_key] = to_be_updated
        return None, result_data, []