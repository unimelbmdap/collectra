from collectra.tasks.base import Task
import importlib

class TaskManager:

    @staticmethod
    def build(task: dict) -> Task:
        """
        Build a Task from a dictionary.
        :param task: The task dictionary containing the task details.
        :raises ValueError: If the task format is invalid or the task type is not recognized
        :return: An instance of Task.
        """
        if not task.get("name") or task["name"] == "":
            raise ValueError("Task name cannot be empty.")
        if not task.get("type"):
            raise ValueError("Task type is required.")
        module_name, class_name = task["type"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls.build(**task)