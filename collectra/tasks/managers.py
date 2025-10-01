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

from collectra.tasks.base import Task
import importlib

class TaskManager:
    """Factory class for creating task instances from configuration dictionaries.
    
    Provides static methods to dynamically instantiate task objects based on
    their type specifications and configuration parameters.
    """

    @staticmethod
    def build(task: dict) -> Task:
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
        if not task.get("name") or task["name"] == "":
            raise ValueError("Task name cannot be empty.")
        if not task.get("type"):
            raise ValueError("Task type is required.")
        module_name, class_name = task["type"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)                     
        return cls.build(**task)