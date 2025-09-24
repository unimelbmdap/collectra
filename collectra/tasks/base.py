from collectra.utils import from_dir, unzip
from dataclasses import dataclass, field, fields
from pathlib import Path
import yaml, ray

LIST_TYPE_FIELDS = ["input", "output"]
DICT_TYPE_FIELDS = ["variables", "params"]

@dataclass(kw_only=True)
class Task:
    name: str
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    config: dict[str, str] = field(default_factory=dict)

    def input_type(self) -> type | tuple:
        raise NotImplementedError("Subclasses must implement this method.")

    def output_type(self) -> type | tuple:
        raise NotImplementedError("Subclasses must implement this method.")

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
        input_data: dict[str, dict | str] = self.check_kwargs(**kwargs)                
        return self.run(**input_data)
    
    def check_file(self, file: str = "") -> tuple[list[str], dict[str, dict | str]]:
        """Check if a file is provided. If so, validate the file format and extract input data.
        
        If no file is provided, return the pending inputs as is.
        If a file is provided, populate the input data from the file and set pending inputs to empty as all inputs are expected to be in the file.

        Args:
            file: The file path to check.
        
        Returns:
            A tuple containing a list of pending input definitions and a dictionary of input data.
        
        """
        pending_inputs = self.input.copy()
        input_data: dict[str, str|dict] = dict()        
        if not file:
            return pending_inputs, input_data
        file_path = Path(file)
        if self.config.get("format", "") != file_path.suffix.replace(".", ""):
            raise ValueError(f"File format {file_path.suffix} does not match expected format {self.config.get('format', '')}")
        data = unzip(file_path, "results.yaml") if file_path.is_file() else from_dir(file_path, "results.yaml") if file_path.is_dir() else dict()                            
        for key in pending_inputs:                        
            input_data[key] = data.get(key, dict())
            if "path" in input_data[key]:
                input_data[key]["path"] = file_path / input_data[key]["path"]
        pending_inputs = []                                     
        return pending_inputs, input_data
    
    def check_kwargs(self, **kwargs) -> dict[str, str|dict]:
        """ Check if the kwargs match the input definitions of the task
        
        If a "file" key is provided, it will validate whether the required inputs are present.
        If there are remaining input definitions not accounted for, it will check the arguments.

        Args: 
            kwargs: The keyword arguments to check. 
        
        Raises: 
            ValueError: If the number of arguments or argument names do not match the input definitions

        """         
        pending_input, input_data = self.check_file(kwargs.pop("file", ""))             
        for key in pending_input:
            if key not in kwargs.keys():
                raise ValueError(f"Missing input for: {key}. Either provide a valid 'file' or the required inputs as arguments.")
            input_data[key] = {'path': str(kwargs[key])}
        input_data.update(kwargs)        
        return input_data
