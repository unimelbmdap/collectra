"""Base task class for Collectra workflows.

This module defines the fundamental task interface used throughout the Collectra system.
It provides the base classes that all tasks must inherit from and implements the core task execution lifecycle.

Classes:
    Task: Base class for all workflow tasks
"""

__all__ = ["Task", "TaskNode"]

import re
from dataclasses import dataclass
from itertools import product
from typing import Generic

from collectra.commons import BaseEntity, Node, NodeStatus, T, TaskContext


class Task(BaseEntity, Generic[T]):
    """Abstract base class for all tasks in Collectra workflows.

    Defines the common interface and behavior that all tasks must implement.

    Attributes:
        name (str): Unique identifier for the task within the workflow.
        input (list[str]): List of input parameter names.
        output (list[str]): List of output names.
        context (TaskContext): Runtime context for task execution.
    """

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name)
        self.context: TaskContext = TaskContext()
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
        input_keys = list(input_dict.keys())
        for parent in parents:
            if parent.name in input_dict:
                input_dict[parent.name].extend(parent.items.values())

        # Filter out empty inputs (READY but no items)
        non_empty_dict = {k: v for k, v in input_dict.items() if v}
        if not non_empty_dict:
            return []
        non_empty_keys = list(non_empty_dict.keys())
        value_lists = list(non_empty_dict.values())
        entries = [list(combination) for combination in list(product(*value_lists))]

        entries_to_remove = []
        for entry_id, entry in enumerate(entries):
            entry_items_by_key = {f"{item.name}": item for item in entry}
            is_valid = True
            for item in entry:
                for parent_id in item.parents:
                    parent_key = re.sub("-.*", "", parent_id)
                    if (
                        parent_key in non_empty_keys
                    ):  # Only validate against non-empty inputs
                        if (
                            parent_key not in entry_items_by_key
                            or entry_items_by_key[parent_key].id != parent_id
                        ):
                            is_valid = False
                            break
            if not is_valid:
                entries_to_remove.append(entry_id)

        entries = [
            entry for id, entry in enumerate(entries) if id not in entries_to_remove
        ]
        return entries

    def run(self, *args) -> T | None:
        """Run the task execution logic.

        This method contains the core task execution logic and must be
        implemented by all subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_output_name(self) -> str:
        name = (
            f"{self.get_name()}_output"
            if not hasattr(self, "output")
            else self.output[0] if isinstance(self.output, list) else self.output
        )
        return name

    def __call__(self, *args) -> T | None:
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
