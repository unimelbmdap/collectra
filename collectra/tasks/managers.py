"""Task management and factory classes for Collectra workflows.

This module provides utilities for dynamically creating and managing task
instances from configuration dictionaries. It handles the instantiation
of tasks based on their type specifications and configuration parameters.

The module includes:
    - Dynamic task instantiation from string class paths
    - Configuration validation and preprocessing
    - Factory pattern implementation for task creation

Classes:
    TaskManager: Factory class for creating task instances from configuration
"""

__all__ = ["TaskManager"]

from typing import get_args

from .base import Task
from .machine_learning import MachineLearningTask
from collectra.utils import load_class_from_string
from utils.get_types import unpack_types, get_param_types


class TaskManager:
    """Factory class for creating task instances from configuration dictionaries.

    Provides static methods to dynamically instantiate task objects based on
    their type specifications and configuration parameters.
    """

    @staticmethod
    def build(task: dict) -> Task:  # This should just be a 'node'
        """Build a Task instance from a configuration dictionary.

        Dynamically loads the task class based on the 'type' field and
        instantiates it with the provided configuration parameters.

        Args:
            task (dict): Configuration dictionary containing task details including
                        'name', 'type', and other task-specific parameters.

        Returns:
            Task: An instantiated task object of the specified type.

        Raises:
            ValueError: If the task name is empty or type is not specified.
            ImportError: If the specified module cannot be imported.
            AttributeError: If the specified class does not exist in the module.
        """
        assert task.get("type"), "Task type is required."
        cls = load_class_from_string(task.pop("type"))
        name = task.pop("name")
        return cls(name=name, **task)

    @staticmethod
    def prepare(task: Task, **kwargs) -> dict:
        """Prepare inputs for the task execution.

        This method can be extended to include input validation, preprocessing,
        or transformation logic as needed.

        Args:
            task (Task): The task instance for which inputs are being prepared.
        """
        breakpoint()
        for flag, item in kwargs.items():
            if flag == "file":
                pass

    @staticmethod
    def prepare_train(task: MachineLearningTask, **kwargs) -> dict:
        """Prepare inputs for the task training.

        This method can be extended to include input validation, preprocessing,
        or transformation logic as needed.

        Args:
            task (Task): The task instance for which training inputs are being prepared.
            inputs (dict): A dictionary of input parameters for the task.
        """
        # This assumes the training function has only one defined type annotation
        type_ = list(unpack_types(task.train, get_param_types).values())[0]
        breakpoint()
