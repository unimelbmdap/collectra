"""Base task classes and execution framework for Collectra workflows.

This module defines the fundamental task interface and execution framework
used throughout the Collectra system. It provides the base classes that all
tasks must inherit from and implements the core task execution lifecycle.

The module includes:
    - Abstract Task base class with input/output management
    - TaskResult class for execution result handling
    - Configuration management and validation
    - File-based input loading from workflows

Classes:
    Task: Abstract base class for all workflow tasks
    TaskResult: Container for task execution results and error handling
"""

__all__ = ["Task", "TaskResult"]

from collectra.utils import from_dir, unzip
from collectra.parsing import load_class_from_string
from dataclasses import dataclass, field, fields
from pathlib import Path
import copy, traceback

LIST_TYPE_FIELDS = ["input", "output"]
DICT_TYPE_FIELDS = ["variables", "params"]

@dataclass(kw_only=True)
class Task:
    """Abstract base class for all tasks in Collectra workflows.
    
    Defines the common interface and behavior that all tasks must implement.
    Handles input/output management, configuration, and execution lifecycle.
    
    Attributes:
        name (str): Unique identifier for the task within the workflow.
        input_keys (list[str]): List of input parameter names.
        input (dict): Dictionary of input data for the task.
        output (dict): Dictionary of output data produced by the task.
        config (dict[str, str]): Configuration parameters for the task.
    """

    name: str
    input_keys: list[str] = field(default_factory=list)
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict )
    config: dict[str, str] = field(default_factory=dict)

    def input_type(self) -> type | tuple:
        """Define the expected input types for this task.
        
        Returns:
            type | tuple: The expected input type(s) for the task.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def output_type(self) -> type | tuple:
        """Define the expected output types for this task.
        
        Returns:
            type | tuple: The expected output type(s) for the task.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def build(cls, **kwargs) -> "Task":
        """Build a task instance from keyword arguments.
        
        Factory method that creates a task instance by parsing the provided
        keyword arguments and organizing them into the appropriate task fields
        (input, output, config, etc.).
        
        Args:
            **kwargs: Keyword arguments containing task configuration and parameters.
            
        Returns:
            Task: A new instance of the task class.
        """             
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

    def __str__(self) -> str:
        """Return string representation of the task.
        
        Returns:
            str: Task name and class name formatted as 'name of ClassName'.
        """
        return f"{self.name} of {self.__class__.__name__}"

    def __repr__(self) -> str:
        """Return string representation of the task for debugging.
        
        Returns:
            str: Task name and class name formatted as 'name of ClassName'.
        """
        return f"{self.name} of {self.__class__.__name__}"

    def set_config(self, config: dict) -> None:
        """Set the configuration for the task.
        
        Updates the task's configuration dictionary with the provided parameters.
        
        Args:
            config (dict): A dictionary containing configuration parameters.
            
        Raises:
            ValueError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise ValueError("Invalid config type.")
        self.config.update(config)

    def run(self) -> dict:
        """Run the task execution logic.
        
        This method contains the core task execution logic and must be
        implemented by all subclasses.
        
        Returns:
            dict: Dictionary containing the task execution results.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")    

    def filter_unused(self) -> dict:
        """Filter input data to only include keys defined in input_keys.
        
        Creates a deep copy of input data containing only the keys that are
        defined in the task's input_keys list.
        
        Returns:
            dict: Filtered dictionary containing only the relevant input data.
        """        
        input_data = dict()
        for key in self.input_keys:
            input_data[key] = copy.deepcopy(self.input.get(key, None))
        return input_data

    def __call__(self, **kwargs) -> "TaskResult":
        """Execute the task with the provided arguments.
        
        Validates input arguments, runs the task execution logic, and returns
        a TaskResult containing the combined input and output data.
        
        Args:
            **kwargs: Keyword arguments for task execution.
            
        Returns:
            TaskResult: Result object containing task execution output.
        """
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
        
        raise NotImplementedError
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
            # required_type = self.input_type()            
            # data = kwargs[key]
            # if not isinstance(data, required_type):
            #     kwargs[key] = required_type(data)        
        self.input.update(kwargs)    

    def save(self, output: dict, output_path: Path, **kwargs) -> tuple[Path, dict, list[Path]]:
        """Save task output to the specified path.
        
        Saves the task's output data to the given output path. Must be
        implemented by subclasses based on their specific output format.
        
        Args:
            output (dict): Dictionary containing the output data to save.
            output_path (Path): Path where the output should be saved.
            **kwargs: Additional keyword arguments for saving.
            
        Returns:
            tuple[Path, dict, list[Path]]: Tuple containing the output path,
                                         saved data dictionary, and list of created file paths.
                                         
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")              

@dataclass(kw_only=True)             
class TaskResult:
    """Container for task execution results and error handling.
    
    Holds the results of task execution including output data, task reference,
    and execution status. Provides methods for creating successful and failed
    result instances.
    
    Attributes:
        output (dict): Dictionary containing the task output data.
        task (Task): Reference to the task that produced this result.
        status (str): Execution status ('successful' or 'failed').
    """

    output: dict
    task: Task
    status: str = "successful"

    def __str__(self) -> str:
        """Return string representation of the task result.
        
        Returns:
            str: The name of the task that produced this result.
        """
        return self.task.name

    def __repr__(self) -> str:
        """Return string representation of the task result for debugging.
        
        Returns:
            str: The name of the task that produced this result.
        """
        return self.task.name

    @staticmethod
    def create_successful(task: Task, output: dict) -> "TaskResult":
        """Create a successful task result.
        
        Factory method for creating a TaskResult instance representing
        successful task execution.
        
        Args:
            task (Task): The task that executed successfully.
            output (dict): The output data produced by the task.
            
        Returns:
            TaskResult: A successful TaskResult instance.
        """
        return TaskResult(output=output, task=task, status="successful")

    @staticmethod
    def create_failed(task: Task, error: Exception) -> "TaskResult":
        """Create a failed task result.
        
        Factory method for creating a TaskResult instance representing
        failed task execution with error information.
        
        Args:
            task (Task): The task that failed to execute.
            error (Exception): The exception that caused the failure.
            
        Returns:
            TaskResult: A failed TaskResult instance with error details.
        """
        result = TaskResult(output=dict(), task=task, status="failed")
        result._set_error(error)
        return result

    def _set_error(self, error: Exception) -> None:
        """Set error information in the task result output.
        
        Stores error details including type, message, and stack trace
        in the output dictionary for debugging purposes.
        
        Args:
            error (Exception): The exception to store information about.
        """        
        self.output["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "stack": traceback.format_exc()
        }
    
    def get_error(self) -> str:
        """Get the error stack trace if the task failed.
        
        Returns the full stack trace of the error that caused the task
        to fail, or an empty string if the task was successful.
        
        Returns:
            str: The error stack trace, or empty string if no error.
        """
        if self.status != "failed" or "error" not in self.output:
            return ""
        return f"{self.output['error']['stack']}"   