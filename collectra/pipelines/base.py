"""Core pipeline execution and workflow management for Collectra.

This module contains the main Collectra workflow class.

It provides the core functionality for defining, excuting and managing 
workflows with task dependencies.

The module supports:
    - Workflow definition and configuration management
    - Task dependency graph construction and execution    
    - Workflow visualization and rendering
    - Training and inference execution modes

Classes:
    Collectra: Main workflow management class    
"""

__all__ = ["Collectra"]

import logging, yaml
import networkx as nx

from pathlib import Path

from collectra import Task
from collectra.utils import load_class_from_string

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def str_presenter(dumper, data):
    """Represent multi-line strings using block style |"""
    if "\n" in data:  # only use | when the string has newlines
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

yaml.add_representer(str, str_presenter)

class Collectra:

    def __init__(self, name: str, ext: str, version: str, path: str = "", **kwargs):
        """Initialize the Collectra workflow.

        Args:
            name (str): Name of the workflow.            
            ext (str): File extension that the workflow accepts.
            version (str): Version of the workflow.
            flow (nx.DiGraph): Acyclic directed graph representing the workflow.
            path (str, optional): Path to the workflow files. Defaults to name.            
            **kwargs: Additional configuration parameters.
        """
        self.name: str = name
        self.ext: str = ext
        self.version: str = version
        self.flow: nx.DiGraph = nx.DiGraph()                
        self.path: Path = Path(name) if not path else Path(path)        
        self.data: dict = kwargs

    def task(self, task_name: str, build: bool = False) -> dict | Task:
        """Get a task from the workflow by name.

        Args:
            task_name (str): Name of the task to retrieve.
        
        Returns:
            dict | None: Task configuration dictionary or None if not found.
        """
        data = self.data.get(task_name, None)
        assert data, f"Task {task_name} not found in workflow"
        if build:
            type_ = data.pop("type", None)
            assert type_ is not None, f"Invalid type  {type_} for {task_name}"
            cls = load_class_from_string(type_)
            task = cls(name=task_name, **data)            
            return task
        return data

    def run(self, task_name: str, **kwargs):        
        task = self.task(task_name, build=True)
        assert isinstance(task, Task)
        result = task(**kwargs)