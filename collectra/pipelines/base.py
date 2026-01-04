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

import copy
import datetime
import json
import logging
import os
import shutil
import time
from pathlib import Path

import graphviz
import networkx as nx
import yaml
from rich import print
from ultralytics.utils.metrics import DetMetrics

from collectra.utils import change_dir, load_class_from_string, remove_exif
from utils.get_types import get_param_types, get_return_type, unpack_types

from ..tasks.base import (
    Task,
    TaskNode,
)
from ..tasks.machine_learning import MachineLearningTask
from ..types.base import (
    Data,
    DataNode,
    Node,
    NodeStatus,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def str_presenter(dumper, data):
    """Represent multi-line strings using block style |"""
    if "\n" in data:  # only use | when the string has newlines
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


class Collectra:

    def __init__(
        self, name: str, ext: str, version: str, path: str | Path = "", **kwargs
    ):
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

    def _check_is_root(self, node_name: str) -> bool:
        """Check if a node is a root node (no parents or only data parents)."""
        parents = list(self.flow.predecessors(node_name))
        for parent in parents:
            grand_parents = list(self.flow.predecessors(parent))
            if len(grand_parents) > 0:
                return False
            parent_node = self._resolve_node(parent)
            if isinstance(parent_node, Task):
                return False
        return True

    def run(self, task_name: str = "", **kwargs):
        """Runs the workflow from the specified task or from all root tasks."""
        # Initialize the workflow if not already done so
        if self.flow.number_of_nodes() == 0:
            self.connect()
        starting_nodes: list = list()
        if task_name:
            task_node = self._resolve_node(task_name)
            assert isinstance(
                task_node, TaskNode
            ), f"Task {task_name} not found in workflow"
            starting_nodes.append(task_node)
        else:
            task_nodes = [
                node["node"]
                for node in self.flow.nodes.values()
                if isinstance(node["node"], TaskNode)
            ]
            for task_node in task_nodes:
                is_root = self._check_is_root(task_node.name)
                if is_root:
                    starting_nodes.append(task_node)
        single = kwargs.pop("single", False)
        files = kwargs.pop("files", [])
        verbose = kwargs.pop("verbose", False)
        usage = kwargs.pop("usage", False)
        render = kwargs.pop("render", False)
        for value in files:
            key = "file" if Path(value).suffix == f".{self.ext}" else "specimen_sheet"
            key, value = self._create_collectra_file(key, value)
            input = {key: value}
            self._init_input_data(starting_nodes, **input, verbose=verbose)
            self.render("workflow_run", file=str(value), render=render)
            if usage:
                os.environ["USAGE_FILE"] = str(
                    Path.cwd() / str(Path(value) / "usage.yaml")
                )
            self._run_nodes(
                starting_nodes, single=single, key=key, value=value, render=render
            )
            os.environ["USAGE_FILE"] = ""

    def _create_collectra_file(
        self, key: str, value: str | Path
    ) -> tuple[str, str | Path]:
        if not key or not value:
            raise ValueError("Both key and value must be provided for input data.")
        savef = Path(value)
        if not savef.exists():
            raise FileNotFoundError(f"Path {savef} does not exist.")
        if savef.is_dir() and (savef / "results.yaml").exists():
            # remove exif data from images in the directory
            for img_file in savef.glob("*"):
                if img_file.suffix.lower() in [
                    ".jpeg",
                    ".jpg",
                    ".png",
                    ".bmp",
                    ".tiff",
                ]:
                    remove_exif(img_file, img_file)
            return key, value
        if savef.is_file() and savef.suffix in [
            ".jpeg",
            ".jpg",
            ".png",
            ".bmp",
            ".tiff",
        ]:
            file_path = savef
            savef = savef.parent / Path(
                savef.name.replace(savef.suffix, f".{self.ext}")
            )
            savef.mkdir(parents=True, exist_ok=True)
            remove_exif(file_path, file_path)
            shutil.copy(file_path, savef / file_path.name)
        if savef.is_dir():
            with change_dir(savef):
                results = dict()
                results["collectra_results_metadata"] = {
                    "workflow": self.name,
                    "version": self.version,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                results[key] = {
                    "type": "collectra.Image",
                    "id": f"{key}",
                    "data": value.name if isinstance(value, Path) else str(value),
                }
                resultsf = Path("results.yaml")
                with open(resultsf, "w") as f:
                    for file_key, data in results.items():
                        yaml.dump(
                            {file_key: data}, f, sort_keys=False, allow_unicode=True
                        )
                        f.write("\n")
        return "file", savef

    def _populate_active_paths(self, nodes: list[TaskNode | DataNode], **kwargs):
        for node in nodes:
            if isinstance(node, DataNode):
                value = kwargs.get(node.name, None)
                node.process(node.name, value, **kwargs)
            children = list(self.flow.successors(str(node.name)))
            children = [self._resolve_node(child) for child in children]
            self._populate_active_paths(children, **kwargs)

    def _init_input_data(self, starting_nodes: list[TaskNode | DataNode], **kwargs):
        self._reset_nodes()
        self.connect()
        for node in starting_nodes:
            parents = self._get_parents_data(node)
            for parent in parents:
                value = kwargs.get(parent.name, None)
                parent.process(parent.name, value, **kwargs)
        self._populate_active_paths(starting_nodes, **kwargs)

    def _reset_nodes(self):
        # Reset all data nodes in the workflow
        nodes = [node["node"] for node in self.flow.nodes.values()]
        for node in nodes:
            if isinstance(node, DataNode):
                node.items = dict()
                node.status = NodeStatus.NOT_READY
            if isinstance(node, TaskNode):
                node.status = NodeStatus.NOT_READY

    def save_run(self, key: str, value: str | Path, data_node: DataNode | None = None):

        if not key or not value:
            print("[yellow]No save location specified, skipping save.[/yellow]")
            return

        savef = Path(value)

        data_nodes = (
            [
                node["node"]
                for node in self.flow.nodes.values()
                if isinstance(node["node"], DataNode)
            ]
            if data_node is None
            else [data_node]
        )

        if data_node is not None and not data_node.status == NodeStatus.READY:
            print(f"[yellow]No new data for {data_node.name}, skipping save.[/yellow]")
            return

        if savef.is_dir():
            with change_dir(savef):
                results = dict()
                resultsf = Path("results.yaml")
                if resultsf.exists():
                    with open(resultsf, "r") as f:
                        results = yaml.safe_load(f)
                    results["collectra_results_metadata"][
                        "timestamp"
                    ] = datetime.datetime.now().isoformat()
                else:
                    results["collectra_results_metadata"] = {
                        "workflow": self.name,
                        "version": self.version,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    results[key] = {
                        "type": "collectra.Image",
                        "data": value.name if isinstance(value, Path) else str(value),
                    }

                for data_node in data_nodes:
                    if data_node.status != NodeStatus.READY:
                        continue
                    name = data_node.name
                    if name in results:
                        results[name] = None
                    for item in data_node.items.values():
                        item_data = item.serialize()
                        if name not in results or results[name] is None:
                            results[name] = list()
                        results[name].append(item_data)
                    if len(results[name]) == 1:
                        results[name] = results[name][0]

                with open("results.yaml", "w") as f:
                    for file_key, data in results.items():
                        yaml.dump(
                            {file_key: data}, f, sort_keys=False, allow_unicode=True
                        )
                        f.write("\n")

        if data_node:
            print(
                f"Results saved to [green]{savef}[/green] for [blue]{data_node.name}[/blue]"
            )
        else:
            print(f"All results saved to [green]{savef}[/green]")

    def _resolve_node(self, node_name: str) -> TaskNode | DataNode:
        node: dict | None = self.flow.nodes.get(node_name, None)
        assert node, f"{node_name} not found in workflow"
        data: TaskNode | DataNode | None = node.get("node", None)
        assert data, f"data for {node_name} not found in workflow. Possible empty node."
        return data

    def _get_parents_data(self, node: Node) -> list[DataNode]:
        parents: list[DataNode] = list()
        parent_names = list(self.flow.predecessors(str(node.name)))
        for parent_name in parent_names:
            parent_node = self._resolve_node(parent_name)
            if isinstance(parent_node, DataNode):
                parents.append(parent_node)
        return parents

    def _get_children_data(self, node: Node) -> list[DataNode]:
        return self._get_data_nodes(list(self.flow.successors(str(node.name))))

    def _get_data_nodes(self, node_names: list[str]) -> list[DataNode]:
        nodes: list[DataNode] = list()
        for node_name in node_names:
            node = self._resolve_node(node_name)
            if isinstance(node, DataNode):
                nodes.append(node)
        return nodes

    def _run_nodes(
        self,
        nodes: list[TaskNode | DataNode] | list[TaskNode] | list[DataNode],
        *args,
        **kwargs,
    ):
        single_run = kwargs.get("single", False)
        render = kwargs.get("render", False)
        child_tasks_of_data_nodes = dict()
        for node in nodes:
            graph_node = self.flow.nodes[str(node.name)]
            graph_node["color"] = "orange"
            graph_node["fontcolor"] = "black"
        self.render("workflow_run", file=kwargs.get("value", ""), render=render)
        for node in nodes:
            if isinstance(node, TaskNode):
                if not self._check_task_ready(node) == NodeStatus.READY:
                    print(f"[yellow]Task {node.name} is not ready, skipping.[/yellow]")
                    continue
                children = list(self.flow.successors(str(node.name)))
                children = [self._resolve_node(child) for child in children]
                results: list = self._run_task(node, **kwargs)
                self.flow.nodes[str(node.name)]["color"] = "green" if results else "red"
                self.flow.nodes[str(node.name)]["fontcolor"] = "black"
                self.render("workflow_run", file=kwargs.get("value", ""), render=render)
                self._run_nodes(children, *results, **kwargs)
            elif isinstance(node, DataNode):
                for arg in args:
                    if not isinstance(arg, Data):
                        continue
                    if node.name == arg.get_name() and node.check_type(type(arg)):
                        self._check_existing(node, arg)
                self.save_run(
                    kwargs.get("key", ""), kwargs.get("value", ""), data_node=node
                )
                self.flow.nodes[str(node.name)]["color"] = "green"
                self.flow.nodes[str(node.name)]["fontcolor"] = "black"
                self.render("workflow_run", file=kwargs.get("value", ""), render=render)
                children = list(self.flow.successors(str(node.name)))
                children = [self._resolve_node(child) for child in children]
                for child in children:
                    if not isinstance(child, TaskNode):
                        continue
                    if child.name not in child_tasks_of_data_nodes and not single_run:
                        child_tasks_of_data_nodes[child.name] = child

        child_tasks = self._check_tasks_ready(list(child_tasks_of_data_nodes.values()))
        if child_tasks:
            self._run_nodes(child_tasks, *args, **kwargs)

    def _check_task_ready(self, task_node: TaskNode) -> NodeStatus:
        parents = self._get_parents_data(task_node)
        if all(parent.status == NodeStatus.READY for parent in parents):
            task_node.status = NodeStatus.READY
        return task_node.status

    def _check_tasks_ready(self, task_nodes: list[TaskNode]) -> list[TaskNode]:
        ready_tasks: list[TaskNode] = list()
        ready_tasks = [
            task_node
            for task_node in task_nodes
            if self._check_task_ready(task_node) == NodeStatus.READY
        ]
        return ready_tasks

    def _check_existing(self, node, new_item: Data) -> None:
        """
        Check if an identical item already exists in the data node.
        If an identical item is found, update the existing item with the new item's ID.
        If no identical item is found, add the new item to the data node.
        """

        def is_identical(original_item: Data, new_item: Data) -> bool:
            same_parents = set(new_item.parents) == set(original_item.parents)

            temp_original_item = copy.deepcopy(original_item).serialize()
            temp_new_item = copy.deepcopy(new_item).serialize()

            temp_original_item.pop("parents", None)
            if isinstance(temp_original_item["data"], str):
                temp_original_item["data"] = temp_original_item["data"].strip().lower()
            if isinstance(temp_new_item["data"], str):
                temp_new_item["data"] = temp_new_item["data"].strip().lower()

            temp_new_item.pop("parents", None)
            temp_new_item.pop("id", None)
            temp_original_item.pop("id", None)
            temp_original_item.pop("parents", None)

            same_attributes = json.dumps(
                temp_original_item, sort_keys=True
            ) == json.dumps(temp_new_item, sort_keys=True)
            return same_parents and same_attributes

        found_identical = False
        items = node.items
        for id in items.keys():
            found_identical = is_identical(items[id], new_item)
            if not found_identical:
                continue
            new_item.id = items[id].id
            items[id] = new_item
            break

        if not found_identical:
            node.add_item(new_item)

    def _run_task(self, task_node: TaskNode, **kwargs) -> list:
        task = task_node.get_task()
        assert isinstance(task, Task), f"Node {task_node.name} is not a Task"
        print(f"Attempting to run task: [blue]{task.name}[/blue]")
        parents = self._get_parents_data(task_node)
        entries = task.prepare_inputs(parents)
        results = list()
        if len(entries) == 0:
            print(
                f"[yellow]No input data found for task {task.name}, skipping.[/yellow]"
            )
            return results
        with change_dir(self.path):
            for entry in entries:
                if not isinstance(entry, list):
                    entry = list(entry)
                entry_result = self._execute_entries(entry, task)
                results.extend(entry_result)
        return results

    def _execute_entries(self, entries: list[Data], task: Task) -> list[Data]:
        try:
            output: Data | list[Data] = task.run(*entries)
            if not isinstance(output, list):
                output = [output]
            for result in output:
                result.set_parents(entries)
            return output
        except Exception as e:
            print(f"[red]Error running task {task.name}: {e}[/red]")
            print("[blue]Affected input data:[/blue]")
            print(entries)
        return list()

    def train(self, task_name: str, **kwargs) -> tuple:
        if self.flow.number_of_nodes() == 0:
            self.connect()
        task_node = self._resolve_node(task_name)
        assert isinstance(
            task_node, TaskNode
        ), f"Task {task_name} not found in workflow"
        task = task_node.get_task()
        assert isinstance(
            task, MachineLearningTask
        ), f"Task {task_name} is not a MachineLearningTask"
        children = self._get_children_data(task_node)
        processed_inputs: list = list()
        inputs = kwargs.pop("input", [])
        kwargs["classes"] = (
            [child.name for child in children]
            if not "classes" in kwargs
            else kwargs["classes"]
        )
        for input in inputs:
            input_path = Path(input)
            item_files = (
                list(input_path.glob(f"*.{self.ext}"))
                if input_path.is_dir()
                else [input_path] if input_path.suffix == self.ext else []
            )
            for item_file in item_files:
                processed_inputs.extend(DataNode.batch_process(item_file, children))
        kwargs = kwargs | self.data.get(task_name, dict()).get("params", dict())
        with change_dir(self.path):
            processed_inputs, validation_results = task.train(
                *processed_inputs, **kwargs
            )
            self.save_train(task, processed_inputs)
        return processed_inputs, validation_results

    def save_train(self, task: MachineLearningTask, results: DetMetrics):
        new_model_path = (
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{task.name}.pt"
        )
        best_model_path = results.save_dir / "weights" / "best.pt"
        if not best_model_path.exists():
            raise Exception(
                f"[red]Best model file not found at {best_model_path}[/red]"
            )
        else:
            print(f"Best model found at: [green]{best_model_path}[/green]")
        task_data = self.data.get(task.name, dict())
        if best_model_path.name != task_data["model"]:
            print(f"Updating model for task {task.name} to {new_model_path}")
            shutil.copy(best_model_path, new_model_path)
            self.data[task.name]["model"] = new_model_path

    @property
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
            new_metadata = self.metadata | self.data
            with open("pipeline.yaml", "w") as f:
                for key, value in new_metadata.items():
                    yaml.dump({key: value}, f, sort_keys=False)
                    f.write("\n")

    def _get_io_list(self, io: list | str) -> list:
        return [io] if isinstance(io, str) else io

    def _check_task_io(self, io_type: str, task: Task, task_config: dict) -> list:
        if io_type not in task_config:
            return list()
        param_types = list(
            unpack_types(
                task.run, get_param_types if io_type == "input" else get_return_type
            ).items()
        )
        io = self._get_io_list(task_config.get(io_type, []))
        nodes: list = list()
        for item in io:
            for _, types in param_types:
                if not isinstance(types, tuple):
                    types = (types,)
                nodes.append((item, types))
        return nodes

    def _check_data_assignment(self, collectra_data: dict, data: dict) -> dict:
        for key, value in data.items():
            if isinstance(value, str) and value in collectra_data:
                data[key] = collectra_data[value]
        return data

    def connect(self):
        """Initialise the DAG representing the workflow. Preparing it for execution."""
        relations: dict[str, dict] = dict()
        with change_dir(self.path):
            for name, value in self.data.items():
                data = copy.deepcopy(value)
                if not isinstance(data, dict) or "type" not in data:
                    continue
                cls_ = load_class_from_string(data.pop("type"))
                data = self._check_data_assignment(self.data, data)
                obj = (
                    self.flow.nodes[name]["node"].get_task()
                    if issubclass(cls_, Task) and self.flow.nodes.get(name, None)
                    else cls_(name, **data)
                )
                if isinstance(obj, Task):
                    task_key = obj.name
                    relations[task_key] = {
                        "input": self._get_io_list(data.get("input", [])),
                        "output": self._get_io_list(data.get("output", [])),
                    }
                    self._add_task_node(task_key, obj)
                    ios: list[tuple[str, list[type]]] = list()
                    ios.extend(self._check_task_io("input", obj, data))
                    ios.extend(self._check_task_io("output", obj, data))
                    for io_key, io_types in ios:
                        self._add_data_node(io_key, types=io_types)
                else:
                    self._add_data_node(name, obj=obj)

            for task_name, io in relations.items():
                inputs = io.get("input", [])
                outputs = io.get("output", [])
                for input in inputs:
                    if self.flow.has_node(input):
                        self.flow.add_edge(input, task_name)
                for output_key in self._get_io_list(outputs):
                    if self.flow.has_node(output_key):
                        self.flow.add_edge(task_name, output_key)

    def _add_task_node(self, name: str, obj: Task):
        node = self.flow.nodes.get(name, None)
        if node:
            node["color"] = "blue"
            node["fontcolor"] = "white"
            return
        node = TaskNode(name, obj)
        self.flow.add_node(
            name,
            node=node,
            label=str(obj),
            shape="box",
            color="blue",
            fontcolor="white",
            style="filled",
        )

    def _add_data_node(
        self, name, obj: Data | None = None, types: list[type] | set[type] = []
    ):
        node = self.flow.nodes.get(name, None)
        if node:
            node["color"] = "salmon"
            node["fontcolor"] = "black"
        if len(types) == 0 and obj:
            types = set([type(obj)])
        if not node:
            items = {obj.id: obj} if obj else {}
            types = set(types)
            node = DataNode(name, items=items, types=types)
            self.flow.add_node(
                name,
                node=node,
                label=str(node),
                shape="oval",
                color="salmon",
                fontcolor="black",
                style="filled",
            )
        elif obj:
            data_node: DataNode = node["node"]
            if not isinstance(data_node, DataNode):
                return
            if type(obj) not in data_node.types:
                return
            data_node.add_item(obj)
            data_node.types = data_node.types.union(set(types))

    def render(self, dest: str | Path = "", file="", render=False) -> None:
        if not render:
            return
        if self.flow.number_of_nodes() == 0:
            self.connect()
        visual_graph: nx.DiGraph = self.flow.copy()
        for node_id in visual_graph.nodes:
            node_data = visual_graph.nodes[node_id]
            if "node" in node_data:
                del node_data["node"]  # Remove the 'node' attribute for visualization
        dot_str: str = nx.nx_pydot.to_pydot(visual_graph).to_string()
        if not dest:
            dest = self.path / f"workflow"
        if file:
            title = f"{self.name} Workflow - Processing: {file}"
            dot_str = dot_str.replace(
                "{", f'{{\nlabel="{title}";\nlabelloc="t";\nfontsize=20;\n\n', 1
            )
        graphviz.Source(dot_str).render(filename=dest, format="svg", cleanup=True)
        time.sleep(0.1)
