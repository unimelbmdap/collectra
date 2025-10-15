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

import logging, yaml, graphviz, copy, datetime, shutil, os
import networkx as nx
from pathlib import Path
from rich.progress import track
from ultralytics.utils.metrics import DetMetrics


from collectra import Task, Data, MachineLearningTask
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

    def __init__(self, name: str, ext: str, version: str, path: str | Path = "", **kwargs):
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
            task = self._resolve_node(node)
            parent_tasks = [task]
        else:
            nodes: list[dict] = [
                node
                for node in self.flow.nodes()
                if len(list(self.flow.predecessors(node))) == 0
            ]
            for node in nodes:
                task = self._resolve_node(node)
                parent_tasks.append(task)
        self._run_nodes(parent_tasks, *args)

    def _resolve_node(self, node: dict) -> Task | Data:
        task: Task | Data | None = node.get("node", None)
        assert task, f"data for {node} not found in workflow. Possible empty node."
        return task

    def _run_nodes(self, tasks: list[Task], *args):
        for task in tasks:
            result: tuple | list = self._run_task(task, *args)
            children = list(self.flow.successors(str(task)))
            if children:
                for child in children:
                    node = self._resolve_node(child)
                    self._run_task(node, *result)

    def _run_task(self, task: Task, *args):
        return task(*args)

    def train(self, task_name: str, **kwargs) -> tuple:
        if self.flow.number_of_nodes() == 0:
            self.connect()        
        node: dict | None = self.flow.nodes.get(task_name, None)
        assert node, f"Task {task_name} not found in workflow"
        task = self._resolve_node(node)
        if not isinstance(task, MachineLearningTask):
            raise ValueError(f"Task {task_name} is not a MachineLearningTask")
        parents = [parent for parent in self.flow.predecessors(task_name)]        
        results: list = list()
        for parent in parents:
            parent_node = self.flow.nodes.get(parent, None) 
            assert parent_node, f"Parent node {parent} not found in workflow"           
            data_node = self._resolve_node(parent_node)
            if not isinstance(data_node, Data):
                continue                        
            for item in kwargs.pop("input", []):
                item_path = Path(item)                
                if item_path.is_dir():
                    for item_file in item_path.glob(f"*.{self.ext}"):                        
                        with change_dir(item_file):
                            result = data_node.handle()                            
                            results.extend(result)
                else:
                    result = data_node.handle(item_path)
                    results.extend(result)
        children = [child for child in self.flow.successors(task_name)]
        if "classes" not in kwargs:
            classes = list()
            for child in children:
                child_node = self.flow.nodes.get(child, None) 
                assert child_node, f"Child node {child} not found in workflow"           
                child_data = self._resolve_node(child_node)
                if not isinstance(child_data, Data):
                    continue
                classes.append(child_data.name)
            kwargs["classes"] = classes             
        kwargs = kwargs | self.data.get(task_name, dict()).get("params", dict())
        with change_dir(self.path):                        
            results, validation_results = task.train(*results, **kwargs)
            self.save_train(task, results)        
        return results, validation_results
    
    def save_train(self, task: MachineLearningTask, results: DetMetrics):        
        new_model_path = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{task.name}.pt"  
        best_model_path = results.save_dir / "weights" / "best.pt"              
        task_data = self.data.get(task.name, dict())
        if best_model_path.name != task_data["model"]:
            shutil.copy(best_model_path, new_model_path)
            if Path(task_data["model"]).exists():
                os.remove(task_data["model"])
            self.data[task.name]["model"] = new_model_path        

    def metadata(self) -> dict:
        return {
            "collectra_pipeline_metadata": {
                "name": self.name,
                "ext": self.ext,
                "version": self.version,
            }
        }

    def save(self):
        with change_dir(self.path):
            new_metadata = self.metadata() | self.data
            with open("pipeline.yaml", "w") as f:
                for key, value in new_metadata.items():
                    yaml.dump({key: value}, f, sort_keys=False)
                    f.write("\n")            

    def _get_task_required_inputs(self, task_config: dict) -> list[str]:
        task_required_input: list | str = task_config.get("input", [])
        if not isinstance(task_required_input, list):
            task_required_input = [task_required_input]
        return task_required_input

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
        """Initialise the DAG representing the workflow. Preparing it for execution."""
        node_dict = dict()
        with change_dir(self.path):
            updated_data_nodes = dict()
            for name, data in track(
                self.data.items(), description="Building workflow..."
            ):
                # Create a copy to avoid modifying original data
                data_copy = copy.deepcopy(data)                
                node_cls = load_class_from_string(data_copy.pop("type"))
                node = node_cls(name, **data_copy)
                node_dict[str(node)] = node
                if isinstance(node, Task):                    
                    task_key = name
                    updated_data_nodes[task_key] = {"input": [], "output": []}
                    self.flow.add_node(
                        task_key,
                        node=node,
                        label=str(node),
                        shape="box",
                        color="blue",
                        fontcolor="white",
                        style="filled",
                    )
                    node_dict[task_key] = node
                    input_nodes = self._check_task_io("input", node, data_copy)
                    output_nodes = self._check_task_io("output", node, data_copy)
                    for io_node in input_nodes:
                        input_key, input_node_item = io_node
                        if input_key not in node_dict:
                            self.flow.add_node(
                                input_key,
                                node=input_node_item,
                                label=str(input_node_item),
                                shape="oval",
                                color="salmon",
                                fontcolor="black",
                                style="filled",
                            )
                            node_dict[input_key] = input_node_item
                        updated_data_nodes[task_key]["input"].append(input_key)
                    for io_node in output_nodes:
                        output_key, output_node_item = io_node
                        if output_key not in node_dict:
                            self.flow.add_node(
                                output_key,
                                node=output_node_item,
                                label=str(output_node_item),
                                shape="oval",
                                color="salmon",
                                fontcolor="black",
                                style="filled",
                            )
                            node_dict[output_key] = output_node_item
                        updated_data_nodes[task_key]["output"].append(output_key)
                else:
                    self.flow.add_node(
                        str(node),
                        node=node,
                        label=str(node),
                        shape="oval",
                        color="salmon",
                        fontcolor="black",
                        style="filled",
                    )
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
            if "node" in node_data:
                del node_data["node"]  # Remove the 'node' attribute for visualization
        dot_str: str = nx.nx_pydot.to_pydot(visual_graph).to_string()
        if not dest:
            dest = self.path / f"workflow"
        graphviz.Source(dot_str).render(filename=dest, format="svg", cleanup=True)
