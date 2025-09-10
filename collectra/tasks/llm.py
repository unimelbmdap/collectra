from collectra.tasks.base import Task
from collectra.images.base import Image
from dataclasses import dataclass, field
from pathlib import Path
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser

@dataclass(kw_only=True)
class CollectraLLM(Task):
    
    model:str
    template:str|Path
    llm:LLM = field(init=False)
    temperature: float = 0.5,
    max_tokens: int = 0,
    variables: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        import llmloader
        
        self.llm = llmloader.load(self.model, temperature=self.temperature, max_tokens=self.max_tokens)

    def replace_in_template(self, prompt, key, value:str|Path|Image):

        if isinstance(value, Image):
            # TODO read and convert to base64
            raise NotImplementedError("Image input not supported in LLM task yet.")

        if Path(value).exists():
            value = Path(value)

        if isinstance(value, Path):
            if value.is_file():
                value = value.read_text()
            else:
                value = str(value)

        return prompt.replace(f"{{{key}}}", str(value))

    def run(self, **kwargs) -> str:
        prompt = str(self.template)
        
        # Replace inputs and config in template
        for key, value in kwargs.items() + self.variables.items():
            prompt = self.replace_in_template(prompt, key, value)
        
        result = self.llm.invoke(prompt)
        parser = StrOutputParser()
        return parser.invoke(result)