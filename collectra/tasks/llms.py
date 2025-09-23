from collectra.tasks.base import Task
from collectra.images.base import Image
from collectra.utils import success_msg, processing_msg
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pathlib import Path
from rich import print
from types import UnionType
import base64, importlib, llmloader

load_dotenv()

@dataclass(kw_only=True)
class LLM(Task):

    name: str
    model: str
    template: str | Path    
    temperature: float = 0.8
    max_tokens: int = 250
    variables: dict = field(default_factory=dict)

    def input_type(self) -> type | tuple:
        return str, Path

    def output_type(self) -> type | tuple:
        return str
    
    def check_kwargs(self, **kwargs) -> dict[str, str|dict|Image]:
        input_data: dict[str, dict | str | Image] = super().check_kwargs(**kwargs)
        for key, value in input_data.items():
            if isinstance(value, dict) and "path" in value and "type" in value:
                module_name, class_name = value.pop("type").rsplit(".", 1)
                cls = getattr(importlib.import_module(module_name), class_name)     
                path = value.pop("path")                
                input_data[key] = cls.build(path, **value)   
        return input_data

    def process_inputs(self, **kwargs) -> tuple:   
        kwargs.update(self.variables)     
        return kwargs.items()

    def __post_init__(self):                                                
        self.llm = AzureChatOpenAI(
            azure_deployment=Path(self.model).name,
            api_version="2024-12-01-preview",
            temperature=self.temperature,
            max_tokens=self.max_tokens            
        )
        # self.llm = llmloader.load(Path(self.model).name, temperature=self.temperature, max_tokens=self.max_tokens)           
        self.messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=self.config.get("system", ""))
        ]    
    
    def image_content(self, image: Image):                
        return {
            "type": "image",
            "source_type": "base64",
            "data": image.get_encoding(),
            "mime_type": image.mime(),
        }

    def add_text(self, prompt: str, key: str, value: str=""):
        return {
            "type": "text",
            "text": prompt.replace(f"{{{key}}}", value).replace("{input}", value).strip()
        }

    def replace_in_template(self, prompt, key, value) -> list[dict | str]:              
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
    
    def run(self, **kwargs) -> dict:                
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
                    results[output_key] = parser.invoke(self.llm.invoke(message))                                    
        print(success_msg(f"[yellow]Inference Complete. Displaying results below: [/yellow]\n\n{results}\n"))        
        return results