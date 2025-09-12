from collectra.tasks.base import Task
from collectra.images.base import Image
from collectra.utils import error_msg
from dataclasses import dataclass, field
from pathlib import Path
from langchain_core.language_models import llms
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import base64

load_dotenv()

@dataclass(kw_only=True)
class LLM(Task):

    name: str
    model: str
    template: str | Path    
    temperature: float = 0.8
    max_tokens: int = 250
    variables: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):        
        import llmloader                
        model_name = Path(self.model).name
        self.llm = llmloader.load(model_name, temperature=self.temperature, max_tokens=self.max_tokens)

    def replace_in_template(self, prompt, key, value:str|Path|Image):

        if isinstance(value, Image):
            # TODO read and convert to base64
            with open(value.image, "rb") as img_file:
                value = base64.b64encode(img_file.read()).decode('utf-8')
                breakpoint()

        if Path(value).exists():
            value = Path(value)

        if isinstance(value, Path):
            if value.is_file():
                value = value.read_text()
            else:
                value = str(value)

        return prompt.replace(f"{{{key}}}", str(value))
    
    def set_config(self, config: dict) -> None:
        """
        Set the configuration for the task.
        :param config: A dictionary containing configuration parameters.
        """
        image_types = [".jpg", ".png"]
        if not isinstance(config, dict):
            raise ValueError("Invalid config type.")
        inputs: ... = config.get("inputs", [])
        for id in range(len(inputs)):
            input = inputs[id]
            input_path = Path(input)
            if input_path.is_file() and input_path.suffix in image_types:
                try:
                    input_image = Image.build(input_path)
                    inputs[id] = input_image
                except Exception as e:
                    print(error_msg(f"couldn't process this input: {e}. Skipping..."))
                    continue

        self.config.update(config)

    def run(self) -> str:
        prompt = str(self.template)
        
        # Replace inputs and config in template
        inputs = self.config.get("inputs", [])
        for input in inputs:   
            self.variables[self.input[0]] = input            
            for key, value in kwargs.items() + self.variables.items():
                prompt = self.replace_in_template(prompt, key, value)
        
        result = self.llm.invoke(prompt)
        parser = StrOutputParser()
        output = parser.invoke(result)
        print(output)
        return output