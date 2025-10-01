from collectra.utils import from_dir, unzip
from dataclasses import dataclass, field, fields
from pathlib import Path
import copy, traceback

LIST_TYPE_FIELDS = ["input", "output"]
DICT_TYPE_FIELDS = ["variables", "params"]

@dataclass(kw_only=True)
class Task:

    name: str
    input_keys: list[str] = field(default_factory=list)
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict )
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
            if cls_field in LIST_TYPE_FIELDS:
                keys = kwargs.pop(cls_field, [])
                iodict = dict()
                for key in keys:
                    iodict[key] = None                    
                input_args[cls_field] = iodict
            elif cls_field in DICT_TYPE_FIELDS:
                input_args[cls_field] = kwargs.pop(cls_field, {})
            else:
                input_args[cls_field] = kwargs.pop(cls_field, None)            
        input_args["config"] = kwargs                 
        return cls(**input_args)

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
            metadata["input"] = self.input if len(self.input) > 1 else self.input[list(self.input.keys())[0]]
        if self.output:
            metadata["output"] = self.output if len(self.output) > 1 else self.output[list(self.output.keys())[0]]
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

    def run(self) -> dict:
        """
        Run the task.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")    

    def filter_unused(self) -> dict:        
        input_data = dict()
        for key in self.input_keys:
            input_data[key] = copy.deepcopy(self.input.get(key, None))
        return input_data

    def __call__(self, **kwargs) -> "TaskResult":
        self.check_kwargs(**kwargs)           
        input_data = self.filter_unused()                                        
        self.run(**input_data)        
        output = self.input | self.output              
        return TaskResult.create_successful(self, output)        

    def check_file(self, file: str = "") -> list[str]:
        """Check if a file is provided. If so, validate the file format and extract input data.
        
        If no file is provided, return the pending inputs as is.
        If a file is provided, populate the input data from the file and set pending inputs to empty as all inputs are expected to be in the file.

        Args:
            file: The file path to check.
        
        Returns:
            A tuple containing a list of pending input definitions and a dictionary of input data.
        
        """                           
        self.input_keys = list(self.input.keys()) 
        pending_keys = self.input_keys.copy()             
        original_input_data = dict()
        for key in self.input:
            if not self.input[key]:
                continue
            original_input_data[key] = copy.deepcopy(self.input[key])        
        self.input.clear()                            
        if not file:
            return pending_keys
        file_path = Path(file)        
        if self.config.get("format", "") != file_path.suffix.replace(".", ""):
            raise ValueError(f"File format {file_path.suffix} does not match expected format {self.config.get('format', '')}")
        data = unzip(file_path, "results.yaml") if file_path.is_file() else from_dir(file_path, "results.yaml") if file_path.is_dir() else dict()                                    
        for key in data:       
            if not data[key]: 
                continue
            if key in pending_keys:
                pending_keys.remove(key)                             
            if "path" in data[key]:
                data[key]["path"] = file_path / data[key]["path"]      
        self.input = data | original_input_data  # Merge with original input to preserve any existing data in        
        return pending_keys            

    def check_kwargs(self, **kwargs) -> None:
        """ Check if the kwargs match the input definitions of the task
        
        If a "file" key is provided, it will validate whether the required inputs are present.
        If there are remaining input definitions not accounted for, it will check the arguments.

        Args: 
            kwargs: The keyword arguments to check. 
        
        Raises: 
            ValueError: If the number of arguments or argument names do not match the input definitions
        """                         
        pending_keys = self.check_file(kwargs.get("file", ""))                                               
        for key in pending_keys:
            if key not in kwargs.keys():
                raise ValueError(f"Missing input for: {key}. Either provide a valid entry file or the required inputs as arguments.")            
        self.input.update(kwargs)    

    def save(self, output: dict, output_path: Path, **kwargs) -> tuple[Path, dict, list[Path]]:
        raise NotImplementedError("Subclasses must implement this method.")              

@dataclass(kw_only=True)             
class TaskResult:

    output: dict
    task: Task
    status: str = "successful"

    def __str__(self) -> str:
        return self.task.name

    def __repr__(self) -> str:
        return self.task.name

    @staticmethod
    def create_successful(task: Task, output: dict) -> "TaskResult":
        return TaskResult(output=output, task=task, status="successful")

    @staticmethod
    def create_failed(task: Task, error: Exception) -> "TaskResult":
        result = TaskResult(output=dict(), task=task, status="failed")
        result._set_error(error)
        return result

    def _set_error(self, error: Exception) -> None:        
        self.output["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "stack": traceback.format_exc()
        }
    
    def get_error(self) -> str:
        if self.status != "failed" or "error" not in self.output:
            return ""
        return f"{self.output['error']['stack']}"   