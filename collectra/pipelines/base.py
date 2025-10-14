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

import logging, yaml, graphviz
import networkx as nx
from pathlib import Path
from rich.progress import track

from collectra import Task, TaskManager, MachineLearningTask
from collectra.utils import load_class_from_string, change_dir
from utils.get_types import get_param_types, get_return_type, unpack_types

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

    def __call__(self, task_name: str, *args):
        self.run(task_name, *args)

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

    def run(self, task_name: str = "", *args):    
        parent_tasks = list()                            
        if self.flow.number_of_nodes() == 0:
            self.connect()
        if task_name:
            node: dict | None = self.flow.nodes.get(task_name, None)        
            assert node, f"Task {task_name} not found in workflow"
            task = self._get_task(node)
            parent_tasks = [task]
        else:
            nodes: list[dict] = [node for node in self.flow.nodes() if len(list(self.flow.predecessors(node))) == 0]
            for node in nodes:
                task = self._get_task(node)
                parent_tasks.append(task)            
        self._run_nodes(parent_tasks, *args)
    
    def _get_task(self, node: dict) -> Task:
        task: Task | None = node.get("node", None)        
        assert task, f"data for {node} not found in workflow. Possible empty node."                
        return task

    def _run_nodes(self, tasks: list[Task], *args):
        for task in tasks:
            result: tuple | list = self._run_task(task, *args)
            children = list(self.flow.successors(str(task)))
            if children:
                for child in children:
                    node = self._get_task(child)
                    self._run_task(node, *result)

    def _run_task(self, task: Task, *args):        
        return task(*args)            

    def train(self, task_name: str, **kwargs):
        task = TaskManager.build(self.task(task_name)) 
        inputs = TaskManager.prepare_train(task, **kwargs)       
        assert isinstance(task, MachineLearningTask)        
        task.train(**inputs)  

    def _get_task_required_inputs(self, task_config: dict) -> list[str]:
        task_required_input: list | str = task_config.get("input", [])
        if not isinstance(task_required_input, list):
            task_required_input = [task_required_input]
        return task_required_input    

    def _check_data_nodes(self, task_config: dict) -> list:
        inputs = list()
        task_required_input: list[str] = self._get_task_required_inputs(task_config)          
        preprocess = task_required_input[0] in self.data        
        while preprocess and len(task_required_input) > 0:                          
            input_item = dict.fromkeys(task_required_input)
            for name in input_item.keys():
                data = self.data.pop(name, None)                
                if isinstance(data, dict):
                    data_cls = load_class_from_string(data.pop("type"))
                    data_item = data_cls(**data)
                    input_item[name] = data_item                        
                else:
                    input_item[name] = None  
            if not any(v is None for v in input_item.values()):
                inputs.append(list(input_item.values()))            
            preprocess = task_required_input[0] in self.data               
        return inputs

    def _check_external_inputs(self, task_config: dict, **kwargs) -> list:
        inputs = list()
        task_required_input: list[str] = self._get_task_required_inputs(task_config)                
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
        
    def _check_input_types(self, param_types: dict | None, inputs: list) -> list:        
        task_inputs = list()        
        if not param_types:
            return inputs
        for input in inputs:
            input_item_dict = dict()
            if len(input) != len(param_types):
                continue
            for index, (param, cls) in enumerate(param_types.items()):
                value = input[index]                
                if not isinstance(value, cls):                                                                        
                    value = cls(*value) if isinstance(value, tuple) else cls(value)                                  
                input_item_dict[param] = value
            task_inputs.append(input_item_dict)        
        return task_inputs

    def _get_io_list(self, io: list | str) -> list:
        if isinstance(io, str):
            io = [io]
        return io

    def _get_io_nodes(self, param_types: list, items: list) -> list:
        """
        Given the parameters of a node, build a list of io nodes.

        If there are options to the parameters, e.g. str | list, then build the node for each option. 
        
        This allows for branching in the workflow.

        """                        
        nodes = list()     
        if len(items) == 0:
            return nodes                 
        for item in items:            
            for param, types in param_types:                                                                             
                if not isinstance(types, tuple):
                    types = (types,)
                for type_ in types:        
                    node = type_(item)                                                                                     
                    nodes.append((str(node), node))
        return nodes

    def _check_task_io(self, io_type: str, task: Task, task_config: dict) -> list:    
        io_list = list()    
        if io_type not in task_config:
            return io_list
        if io_type == "input":
            param_types = list(unpack_types(task.run, get_param_types).items())            
            inputs = self._get_io_list(task_config.get("input", []))                 
            io_list = self._get_io_nodes(param_types, inputs)            
        if io_type == "output":            
            param_types = list(unpack_types(task.run, get_return_type).items())
            outputs = self._get_io_list(task_config.get("output", []))                 
            io_list = self._get_io_nodes(param_types, outputs)
        return io_list

    def connect(self):
        """ Initialise the DAG representing the workflow. Preparing it for execution.
        """
        node_dict = dict()
        with change_dir(self.path):            
            updated_data_nodes = dict()
            for name, data in track(self.data.items(), description="Building workflow..."):
                # Create a copy to avoid modifying original data
                data_copy = data.copy()
                node_cls = load_class_from_string(data_copy.pop("type"))
                node = node_cls(name, **data_copy)             
                node_dict[str(node)] = node                
                if isinstance(node, Task):
                    # Use consistent key - str(node) for task name
                    task_key = name
                    updated_data_nodes[task_key] = {"input": [], "output": []}
                    self.flow.add_node(task_key, node=node, label=str(node), shape="box", color="blue", fontcolor="white", style="filled")
                    node_dict[task_key] = node
                    input_nodes = self._check_task_io("input", node, data_copy)
                    output_nodes = self._check_task_io("output", node, data_copy)                                   
                    for io_node in input_nodes:
                        input_key, input_node_item = io_node
                        if input_key not in node_dict:                            
                            self.flow.add_node(input_key, node=input_node_item, label=str(input_node_item), shape="oval", color="salmon", fontcolor="black", style="filled")                     
                            node_dict[input_key] = input_node_item
                        updated_data_nodes[task_key]["input"].append(input_key)
                    for io_node in output_nodes:
                        output_key, output_node_item = io_node
                        if output_key not in node_dict:                            
                            self.flow.add_node(output_key, node=output_node_item, label=str(output_node_item), shape="oval", color="salmon", fontcolor="black", style="filled")                     
                            node_dict[output_key] = output_node_item
                        updated_data_nodes[task_key]["output"].append(output_key)
                else:                    
                    self.flow.add_node(str(node), node=node, label=str(node), shape="oval", color="salmon", fontcolor="black", style="filled")       
                    node_dict[str(node)] = node                    
            for task_key, data in updated_data_nodes.items():                
                inputs = self._get_io_list(data.get("input", []))
                outputs = self._get_io_list(data.get("output", []))   
                for input in inputs:
                    self.flow.add_edge(input, task_key)
                for output in outputs:
                    self.flow.add_edge(task_key, output)

    def render(self, dest: str | Path = ""):
        visual_graph: nx.DiGraph = self.flow.copy()
        for node_id in visual_graph.nodes:
            node_data = visual_graph.nodes[node_id]
            if 'node' in node_data:
                del node_data['node']  # Remove the 'node' attribute for visualization    
        dot_str: str = nx.nx_pydot.to_pydot(visual_graph).to_string()        
        if not dest:
            dest = self.path / f"workflow"                
        graphviz.Source(dot_str).render(filename=dest, format="svg", cleanup=True)


