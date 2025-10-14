"""Base task class for Collectra workflows.

This module defines the fundamental task interface used throughout the Collectra system.
It provides the base classes that all tasks must inherit from and implements the core task execution lifecycle.

Classes:
    Task: Base class for all workflow tasks
"""

__all__ = ["Task"]

from typing import Generic
from pathlib import Path    

from collectra.commons import Node, T

class Task(Node, Generic[T]):
    """Abstract base class for all tasks in Collectra workflows.

    Defines the common interface and behavior that all tasks must implement.

    Attributes:
        name (str): Unique identifier for the task within the workflow.
        input (list[str]): List of input parameter names.
        output (list[str]): List of output names.
    """

    name: str

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)   

    def get_name(self) -> str:
        """Get the name of the task.

        Returns:
            str: The name of the task.
        """
        return self.name
    
    def serialize(self) -> dict:
        serialized = dict(type=self.get_class_path())
        for key, value in self.attributes.items():
            if isinstance(value, Path):
                value = str(value)
            serialized[key] = value
        return serialized

    def __representation(self) -> str:
        """Return string representation of the task.

        Returns:
            str: Task name and class name formatted as 'name of ClassName'.
        """
        return f"{self.get_name()}\n{self.get_class_path()}"

    def __str__(self) -> str:
        """Return string representation of the task.

        Returns:
            str: Task name and class name formatted as 'name of ClassName'.

        """
        return self.__representation()

    def __repr__(self) -> str:
        """Return string representation of the task.

        Returns:
            str: Task name and class name formatted as 'name of ClassName'.

        """
        return self.__representation()

    def run(self, *args) -> T:
        """Run the task execution logic.

        This method contains the core task execution logic and must be
        implemented by all subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, **kwargs) -> T:
        """Run the task with the provided arguments.

        Args:
            **kwargs: Keyword arguments for task execution.

        Returns:

        """
        return self.run(**kwargs)
        
