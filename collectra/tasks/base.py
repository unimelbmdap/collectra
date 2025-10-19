"""Base task class for Collectra workflows.

This module defines the fundamental task interface used throughout the Collectra system.
It provides the base classes that all tasks must inherit from and implements the core task execution lifecycle.

Classes:
    Task: Base class for all workflow tasks
"""

__all__ = ["Task", "TaskNode"]

from typing import Generic
from dataclasses import dataclass

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

    def __post_init__(self):
        self._status = NodeStatus.READY if self.task else NodeStatus.NOT_READY
    
    def get_task(self) -> Task:
        return self.task   