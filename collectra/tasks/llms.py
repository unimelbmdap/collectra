from collectra.tasks.base import Task
from collectra.images.base import Image
from collectra.utils import success_msg
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from rich import print
import llmloader
import base64

load_dotenv()

@dataclass(kw_only=True)
class LLM(Task):

    name: str
    model: str
    template: str | Path    
    temperature: float = 0.8
    max_tokens: int = 250
    variables: dict = field(default_factory=dict)

    def input_type(self) -> type:
        return str | Path | Image

    def output_type(self) -> type:
        return str

    def __post_init__(self):                                        
        self.llm = llmloader.load(Path(self.model).name, temperature=self.temperature, max_tokens=self.max_tokens)           
        self.messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=self.config.get("system", ""))
        ]
    
    def is_image(self, value: Path):
        return value.is_file(), value.suffix in [".jpg", ".png"]
    
    def encode_image(self, image_path: Path):
        types = {
            ".jpg": "jpeg",
            ".png": "png",
        }
        with open(image_path, "rb") as img_file:
            value = base64.b64encode(img_file.read()).decode('utf-8')                       
        return {
            "type": "image",
            "source_type": "base64", 
            "data": value,
            "mime_type": f"image/{types[image_path.suffix]}"
        }  
    
    def add_text(self, prompt: str, key: str, value: str =""):
        return {
            "type": "text",
            "text": prompt.replace(f"{{{key}}}", value).strip()
        }

    def replace_in_template(self, prompt, items) -> list[dict | str]:              
        content = []
        for key, value in items:
            if isinstance(value, Image):                  
                content.append(self.add_text(prompt, key))      
                content.append(self.encode_image(Path(value.image)))                                 
            elif Path(value).exists():                
                image_path = Path(value)
                is_file, is_image = self.is_image(image_path)
                if is_file and is_image:
                    content.append(self.add_text(prompt, key))    
                    content.append(self.encode_image(image_path))                     
                elif is_file:
                    value = value.read_text()                    
                    content.append(self.add_text(prompt, key, str(value)))     
            else:
                content.append(self.add_text(prompt, key, str(value)))     
        return content
    
    def run(self, **kwargs) -> str:
        prompt = str(self.template)                   
        content = self.replace_in_template(prompt, kwargs.items() | self.variables.items())                
        self.messages.append(HumanMessage(content=content))         
        result = self.llm.invoke(self.messages)
        parser = StrOutputParser()
        output = parser.invoke(result)
        print(success_msg(f"[yellow]Inference Complete. Displaying results below: [/yellow]\n\n{output}\n"))        
        return output