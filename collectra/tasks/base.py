from dataclasses import dataclass, field, fields
from collectra.utils import processing_msg

LIST_TYPE_FIELDS = ["input", "output"]
DICT_TYPE_FIELDS = ["variables", "params"]

@dataclass(kw_only=True)
class Task:
    name: str
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    config: dict[str, str] = field(default_factory=dict)

    def input_type(self) -> type:
        raise NotImplementedError("Subclasses must implement this method.")

    def output_type(self) -> type:
        return str

    @classmethod
    def build(cls, **kwargs) -> "Task":             
        cls_fields = [f.name for f in fields(cls)]
        input_args = dict()        
        for cls_field in cls_fields:
            if cls_field == "config": 
                continue
            if cls_field in LIST_TYPE_FIELDS:
                input_args[cls_field] = kwargs.pop(cls_field, [])
            elif cls_field in DICT_TYPE_FIELDS:
                input_args[cls_field] = kwargs.pop(cls_field, {})
            else:
                input_args[cls_field] = kwargs.pop(cls_field, None)            
        input_args["config"] = kwargs                
        return cls(**input_args)

    def __post_init__(self):
        if isinstance(self.input, str):
            self.input = [self.input]
        if isinstance(self.output, str):
            self.output = [self.output]

    def metadata(self) -> dict:
        """
        Get metadata of the task.
        :return: A dictionary containing task metadata.
        """
        metadata = {
            "name": self.name,
            "model": self.config.get("model", ""),
        }
        if self.input:
            metadata["input"] = self.input if len(self.input) > 1 else self.input[0]  # type: ignore
        if self.output:
            metadata["output"] = self.output if len(self.output) > 1 else self.output[0]  # type: ignore
        return metadata

    def __str__(self) -> str:
        return f"{self.name} of {self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.name} of {self.__class__.__name__}"

    def set_config(self, config: dict) -> None:
        """
        Set the configuration for the task.
        :param config: A dictionary containing configuration parameters.
        """
        if not isinstance(config, dict):
            raise ValueError("Invalid config type.")
        self.config.update(config)

    def run(self) -> list[dict] | None:
        """
        Run the task.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, **kwargs):
        self.check_kwargs(**kwargs)        
        return self.run(**kwargs)
    
    def check_kwargs(self, **kwargs) -> None:          
        assert len(kwargs) == len(self.input), f"Number of arguments for {self.name} is incorrect. Expected {len(self.input)} and received {len(kwargs)}" # type: ignore
        for key in kwargs.keys():
            if key not in self.input:
                raise ValueError(f"input {key} does not exist in input definitions: {', '.join(self.input)}")        
            
        