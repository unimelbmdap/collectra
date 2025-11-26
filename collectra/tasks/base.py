"""Base task class for Collectra workflows.

This module defines the fundamental task interface used throughout the Collectra system.
It provides the base classes that all tasks must inherit from and implements the core task execution lifecycle.

Classes:
    Task: Base class for all workflow tasks
"""

__all__ = ["Task", "TaskNode"]

from typing import Generic
from dataclasses import dataclass
from itertools import product

from collectra.commons import BaseEntity, T, Node, NodeStatus


class Task(BaseEntity, Generic[T]):
    """Abstract base class for all tasks in Collectra workflows.

    Defines the common interface and behavior that all tasks must implement.

    Attributes:
        name (str): Unique identifier for the task within the workflow.
        input (list[str]): List of input parameter names.
        output (list[str]): List of output names.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def input_dict(self) -> dict[str, list]:     
        input_dict = dict()
        if hasattr(self, "input"):
            if not isinstance(self.input, list):
                input_list = [self.input]       
            else:
                input_list = self.input.copy()
            input_dict = {k: [] for k in input_list}
        return input_dict
    
    def prepare_inputs(self, parents: list) -> list:        
        input_dict = self.input_dict
        for parent in parents:
            if parent.name in input_dict:
                input_dict[parent.name].extend(parent.items.values())    
        value_lists = [inputs for inputs in input_dict.values()]
        all_combinations = list(product(*value_lists))
        entries = [list(combination) for combination in all_combinations]
        return entries

    def run(self, *args) -> T:
        """Run the task execution logic.

        This method contains the core task execution logic and must be
        implemented by all subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, *args) -> T:
        """Run the task with the provided arguments.

        Args:
            **kwargs: Keyword arguments for task execution.

        Returns:

        """
        return self.run(*args)


@dataclass
class TaskNode(Node):

    task: Task    

    def get_task(self) -> Task:
        return self.task
