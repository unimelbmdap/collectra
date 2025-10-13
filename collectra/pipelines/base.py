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

import logging, yaml, os
import networkx as nx

from pathlib import Path

from collectra import Task, TaskManager, MachineLearningTask
from collectra.utils import load_class_from_string
from utils.get_types import get_param_types

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
        self.path: Path = Path.cwd() / name if not path else Path(path)        
        self.data: dict = kwargs

    def __call__(self, task_name: str, **kwargs):
        self.run(task_name, **kwargs)

    def task(self, task_name: str) -> dict:
        """Get a task from the workflow by name.

        Args:
            task_name (str): Name of the task to retrieve.
        
        Returns:
            dict | None: Task configuration dictionary or None if not found.
        """
        data = self.data.get(task_name, None)
        assert data, f"Task {task_name} not found in workflow"        
        data["name"] = task_name
        return data

    def run(self, task_name: str, **kwargs):                
        current_path = Path.cwd()
        os.chdir(self.path)        
        task_config = self.task(task_name)
        inputs = list()
        task = TaskManager.build(task_config)
        data_nodes = self._check_data_nodes(task_config)
        external_inputs = self._check_external_inputs(task_config, **kwargs)
        if data_nodes:
            inputs.append(data_nodes)                
        if external_inputs:
            inputs.append(external_inputs)        
        inputs = self._match_input_types(get_param_types(task.run), inputs)               
        for input in inputs:                               
            task(**input)
        os.chdir(current_path)

    def train(self, task_name: str, **kwargs):
        task = TaskManager.build(self.task(task_name)) 
        inputs = TaskManager.prepare_train(task, **kwargs)       
        assert isinstance(task, MachineLearningTask)        
        task.train(**inputs)  

    def _get_task_required_inptus(self, task_config: dict) -> list[str]:
        task_required_input: list | str = task_config.get("input", [])
        if not isinstance(task_required_input, list):
            task_required_input = [task_required_input]
        return task_required_input    

    def _check_data_nodes(self, task_config: dict) -> list:
        task_required_input: list[str] = self._get_task_required_inptus(task_config)        
        input_item = dict()                           
        for name in task_required_input:
            if name in self.data:
                data = self.data[name]                
                data_cls = load_class_from_string(data.pop("type"))
                data_item = data_cls(**data)
                input_item[name] = data_item                  
        return list(input_item.values())

    def _check_external_inputs(self, task_config: dict, **kwargs) -> list:
        inputs = list()
        task_required_input: list[str] = self._get_task_required_inptus(task_config)                
        required_input_arr_len = len(task_required_input)
        kwargs_inputs = list(kwargs.items())
        index = 0
        while index < len(kwargs_inputs):  
            index_next = index + required_input_arr_len
            if index_next > len(kwargs_inputs):
                break
            else:
                input_item = dict()                
                sub_inputs = kwargs_inputs[index:index_next]
                for sub_index in range(len(task_required_input)):
                    name, value = sub_inputs[sub_index]
                    if name == task_required_input[sub_index]:                                       
                        input_item[name] = value
                if len(input_item) == len(task_required_input):                  
                    inputs.extend(list(input_item.items()))
            index = index_next                       
        return inputs        
        
    def _match_input_types(self, param_types: dict | None, inputs: list) -> list:                
        if not param_types:
            return inputs
        for input_index, input in enumerate(inputs):
            input_item_dict = dict()
            for index, (param, cls) in enumerate(param_types.items()):
                value = input[index]                
                if not isinstance(value, cls):                                        
                    value = cls(*value) if isinstance(value, tuple) else cls(value)                                            
                input_item_dict[param] = value
            inputs[input_index] = input_item_dict        
        return inputs
                            