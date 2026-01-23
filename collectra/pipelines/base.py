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
import shutil
import time
import traceback
from pathlib import Path

import graphviz
import networkx as nx
import yaml
from rich.table import Table
from ultralytics.utils.metrics import DetMetrics

from collectra.utils import change_dir, load_class_from_string, remove_exif
from utils.get_types import get_param_types, get_return_type, unpack_types

from ..commons.base import TaskContext
from ..tasks.base import (
    Task,
    TaskNode,
)
from ..tasks.machine_learning import MachineLearningTask
from ..types.base import (
    Data,
    DataNode,
    NodeStatus,
)
from .node_graph_manager import NodeGraphManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def str_presenter(dumper, data):
    """Represent multi-line strings using block style |"""
    if "\n" in data:  # only use | when the string has newlines
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


class Collectra:
    """Main workflow management class for Collectra pipelines.

    This class orchestrates the execution of complex data processing workflows
    by managing task dependencies, data flow, and execution state.
    """

    # Constants
    IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".tiff")
    RESULTS_FILE = "results.yaml"
    METADATA_KEY = "collectra_results_metadata"
    PIPELINE_METADATA_KEY = "collectra_pipeline_metadata"

    # Graph visualization colors
    COLOR_PROCESSING = "orange"
    COLOR_SUCCESS = "green"
    COLOR_FAILURE = "red"
    COLOR_FONT = "black"

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
        self.node_manager: NodeGraphManager = NodeGraphManager(self.flow)
        self.path: Path = Path.cwd() / name if not path else Path(path)
        self.data: dict = kwargs

    def __call__(self, task_name: str, **kwargs):
        """Allow instance to be called directly to run workflow."""
        self.run(task_name, **kwargs)

    # ==================== Configuration Methods ====================

    def task(self, task_name: str) -> dict:
        """Get a task from the workflow by name.

        Args:
            task_name (str): Name of the task to retrieve.

        Returns:
            dict | None: Task configuration dictionary or None if not found.
        """
        data = self.data.get(task_name, None)
        if not data:
            raise ValueError(f"Task {task_name} not found in workflow")
        data["name"] = task_name
        return data

    # ==================== Node Status and Validation Methods ====================

    def _check_is_root(self, node_name: str) -> bool:
        """Check if a node is a root node (no parents or only data parents)."""
        parents = list(self.flow.predecessors(node_name))
        for parent in parents:
            grand_parents = list(self.flow.predecessors(parent))
            if len(grand_parents) > 0:
                return False
            parent_node = self.node_manager.resolve_node(parent)
            if isinstance(parent_node, Task):
                return False
        return True

    # ==================== Logging Methods ====================

    def create_log_table(self):
        self.log = Table()
        self.log.add_column("Source", justify="left", style="green")
        self.log.add_column("Message", justify="left", style="magenta")
        self.log.add_column("Traceback", justify="left", style="red")
        self.log.add_column("Affected inputs", justify="left", style="blue")

    # ==================== Workflow Execution Methods ====================

    def run(self, task_name: str = "", **kwargs):
        """Runs the workflow from the specified task or from all root tasks."""
        self.create_log_table()
        try:
            self._ensure_workflow_connected()
            starting_nodes = self._get_starting_nodes(task_name)
            run_options = self._extract_run_options(kwargs)
            self._process_files(starting_nodes, run_options)
        except Exception as e:
            self.log.add_row("workflow_run", str(e), traceback.format_exc())

    def _ensure_workflow_connected(self):
        """Ensure the workflow graph is connected before execution."""
        if self.flow.number_of_nodes() == 0:
            self.connect()

    def _get_starting_nodes(self, task_name: str) -> list[TaskNode]:
        """Get the starting nodes for workflow execution.

        Args:
            task_name: Specific task to start from, or empty for all root tasks.

        Returns:
            List of starting task nodes.
        """
        if task_name:
            return self._get_specific_starting_node(task_name)
        return self._get_root_starting_nodes()

    def _get_specific_starting_node(self, task_name: str) -> list[TaskNode]:
        """Get a specific task node as the starting point."""
        task_node = self.node_manager.resolve_node(task_name)
        if not isinstance(task_node, TaskNode):
            raise TypeError(f"Task {task_name} not found in workflow")
        return [task_node]

    def _get_root_starting_nodes(self) -> list[TaskNode]:
        """Get all root task nodes as starting points."""
        starting_nodes = []
        task_nodes = [
            node["node"]
            for node in self.flow.nodes.values()
            if isinstance(node["node"], TaskNode)
        ]
        for task_node in task_nodes:
            if self._check_is_root(task_node.name):
                starting_nodes.append(task_node)
        return starting_nodes

    def _extract_run_options(self, kwargs: dict) -> dict:
        """Extract and return run options from kwargs."""
        return {
            "single": kwargs.pop("single", False),
            "files": kwargs.pop("files", []),
            "usage": kwargs.pop("usage", False),
            "render": kwargs.pop("render", False),
            "output": kwargs.pop("output", None),
        }

    def _process_files(self, starting_nodes: list[TaskNode], options: dict):
        """Process each file in the workflow.

        Args:
            starting_nodes: List of nodes to start execution from.
            options: Execution options (single, files, usage, render, output).
        """
        for file_path in options["files"]:
            try:
                self._process_single_file(starting_nodes, file_path, options)
            except Exception as e:
                self.log.add_row("workflow_run", str(e), traceback.format_exc())

    def _process_single_file(
        self, starting_nodes: list[TaskNode], file_path: str, options: dict
    ):
        """Process a single file through the workflow.

        Args:
            starting_nodes: Nodes to start from.
            file_path: Path to file to process.
            options: Execution options.
        """
        key = "file" if Path(file_path).suffix == f".{self.ext}" else "specimen_sheet"
        key, value = self._create_collectra_file(key, file_path, options["output"])

        # Create task context for this file
        context = TaskContext(
            usage_file=(
                Path.cwd() / str(Path(value) / "usage.yaml")
                if options["usage"]
                else None
            ),
            render=options["render"],
            file_path=Path(value),
            single_run=options["single"],
        )

        # Set context for all tasks
        self._set_task_contexts(context)

        input_data = {key: value}
        self._init_input_data(starting_nodes, **input_data)
        self.render("workflow_run", file=str(value), render=options["render"])

        self._run_nodes(
            starting_nodes,
            single=options["single"],
            key=key,
            value=value,
            render=options["render"],
        )

    # ==================== File Operations ====================

    def _create_collectra_file(
        self, key: str, value: str | Path, output_dir: str | Path | None = None
    ) -> tuple[str, str | Path]:
        """Create or prepare a collectra file structure for processing.

        Handles three cases:
        - Existing collectra folder with results.yaml
        - Single image file
        - Directory without results.yaml
        """
        self._validate_file_inputs(key, value)
        savef = Path(value)

        # Case A: Existing collectra folder with results.yaml
        if self._is_existing_collectra_folder(savef):
            return key, self._handle_existing_folder(savef, output_dir)

        # Case B: Single image file
        if self._is_image_file(savef):
            savef = self._handle_image_file(savef, output_dir)

        # Case C: Directory without results.yaml - create results file
        if savef.is_dir():
            self._create_results_file(savef, key, value)

        return "file", savef

    def _validate_file_inputs(self, key: str, value: str | Path):
        """Validate input parameters for file creation."""
        if not key or not value:
            raise ValueError("Both key and value must be provided for input data.")

        savef = Path(value)
        if not savef.exists():
            raise FileNotFoundError(f"Path {savef} does not exist.")

    def _is_existing_collectra_folder(self, path: Path) -> bool:
        """Check if path is an existing collectra folder with results."""
        return path.is_dir() and (path / self.RESULTS_FILE).exists()

    def _is_image_file(self, path: Path) -> bool:
        """Check if path is an image file."""
        return path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _handle_existing_folder(
        self, savef: Path, output_dir: str | Path | None
    ) -> Path:
        """Handle existing collectra folder, optionally copying to output."""
        if output_dir:
            output_path = Path(output_dir) / savef.name
            output_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(savef, output_path, dirs_exist_ok=True)
            savef = output_path

        self._remove_exif_from_images(savef)
        return savef

    def _remove_exif_from_images(self, directory: Path):
        """Remove EXIF data from all images in directory."""
        for img_file in directory.glob("*"):
            if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                remove_exif(img_file, img_file)

    def _handle_image_file(
        self, file_path: Path, output_dir: str | Path | None
    ) -> Path:
        """Handle single image file by creating collectra folder structure."""
        folder_name = file_path.name.replace(file_path.suffix, f".{self.ext}")
        savef = (
            Path(output_dir) / folder_name
            if output_dir
            else file_path.parent / folder_name
        )
        savef.mkdir(parents=True, exist_ok=True)

        remove_exif(file_path, file_path)
        shutil.copy(file_path, savef / file_path.name)
        return savef

    def _create_results_file(self, directory: Path, key: str, value: str | Path):
        """Create a new results.yaml file in the given directory."""
        with change_dir(directory):
            results = {
                self.METADATA_KEY: {
                    "workflow": self.name,
                    "version": self.version,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                key: {
                    "type": "collectra.Image",
                    "id": f"{key}",
                    "data": value.name if isinstance(value, Path) else str(value),
                },
            }
            self._write_results_file(results)

    # ==================== Data Initialization and Node Reset ====================

    def _populate_active_paths(
        self,
        nodes: list[TaskNode | DataNode] | list[TaskNode] | list[DataNode],
        **kwargs,
    ):
        for node in nodes:
            if isinstance(node, DataNode):
                value = kwargs.get(node.name, None)
                node.process(node.name, value, **kwargs)
                node.catcher.flush_msg(self.log)

            children = list(self.flow.successors(str(node.name)))
            children = [self.node_manager.resolve_node(child) for child in children]
            self._populate_active_paths(children, **kwargs)

    def _init_input_data(
        self,
        starting_nodes: list[TaskNode | DataNode] | list[TaskNode] | list[DataNode],
        **kwargs,
    ):
        self._reset_nodes()
        self.connect()
        for node in starting_nodes:
            parents = self.node_manager.get_parents_data(node)
            for parent in parents:
                value = kwargs.get(parent.name, None)
                parent.process(parent.name, value, **kwargs)
                parent.catcher.flush_msg(self.log)
        self._populate_active_paths(starting_nodes, **kwargs)

    def _set_task_contexts(self, context: TaskContext):
        """Set context for all tasks in the workflow.

        Args:
            context: TaskContext to apply to all tasks.
        """
        for node in self.flow.nodes.values():
            if isinstance(node["node"], TaskNode):
                task = node["node"].get_task()
                if isinstance(task, Task):
                    task.context = context

    def _reset_nodes(self):
        # Reset all data nodes in the workflow
        nodes = [node["node"] for node in self.flow.nodes.values()]
        for node in nodes:
            if isinstance(node, DataNode):
                node.items = dict()
                node.ensemble_items = dict()
                node.status = NodeStatus.NOT_READY
            if isinstance(node, TaskNode):
                node.status = NodeStatus.NOT_READY

    # ==================== Save Operations ====================

    def save_run(self, key: str, value: str | Path, data_node: DataNode | None = None):
        """Save workflow execution results to file.

        Args:
            key: Key identifier for the data.
            value: Path where results should be saved.
            data_node: Specific data node to save, or None for all nodes.
        """
        if not self._should_save(key, value, data_node):
            return

        savef = Path(value)
        if not savef.is_dir():
            return

        data_nodes = self._get_data_nodes_to_save(data_node)
        self._save_results_to_directory(savef, key, value, data_nodes)
        self._log_save_success(savef, data_node)

    def _should_save(
        self, key: str, value: str | Path, data_node: DataNode | None
    ) -> bool:
        """Check if saving should proceed."""
        if not key or not value:
            self.log.add_row(
                "workflow_run", "No save location specified, skipping save."
            )
            return False

        if data_node is not None and data_node.status != NodeStatus.READY:
            self.log.add_row(
                "workflow_run", f"No new data for {data_node.name}, skipping save."
            )
            return False

        return True

    def _get_data_nodes_to_save(
        self, data_node: DataNode | None = None
    ) -> list[DataNode]:
        """Get list of data nodes to save."""
        if data_node is not None:
            return [data_node]

        return [
            node["node"]
            for node in self.flow.nodes.values()
            if isinstance(node["node"], DataNode)
        ]

    def _save_results_to_directory(
        self, directory: Path, key: str, value: str | Path, data_nodes: list[DataNode]
    ):
        """Save results to directory with proper formatting."""
        with change_dir(directory):
            results = self._load_or_create_results(key, value)
            results = self._collect_node_data(results, data_nodes)
            self._write_results_file(results)

    def _load_or_create_results(self, key: str, value: str | Path) -> dict:
        """Load existing results file or create new results dictionary."""
        resultsf = Path(self.RESULTS_FILE)

        if resultsf.exists():
            with open(resultsf, "r") as f:
                results = yaml.safe_load(f)
            results[self.METADATA_KEY][
                "timestamp"
            ] = datetime.datetime.now().isoformat()
        else:
            results = {
                self.METADATA_KEY: {
                    "workflow": self.name,
                    "version": self.version,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                key: {
                    "type": "collectra.Image",
                    "data": value.name if isinstance(value, Path) else str(value),
                },
            }

        return results

    def _collect_node_data(self, results: dict, data_nodes: list[DataNode]) -> dict:
        """Collect data from all ready nodes and add to results."""
        for data_node in data_nodes:
            if data_node.status != NodeStatus.READY:
                continue

            name = data_node.name
            results[name] = self._serialize_node_items(data_node)

        return results

    def _serialize_node_items(self, data_node: DataNode) -> dict | list:
        """Serialize all items in a data node."""
        serialized_items = [item.serialize() for item in data_node.items.values()]
        return serialized_items[0] if len(serialized_items) == 1 else serialized_items

    def _write_results_file(self, results: dict):
        """Write results dictionary to YAML file."""
        with open(self.RESULTS_FILE, "w") as f:
            for file_key, data in results.items():
                yaml.dump({file_key: data}, f, sort_keys=False, allow_unicode=True)
                f.write("\n")

    def _log_save_success(self, savef: Path, data_node: DataNode | None):
        """Log successful save operation."""
        if data_node:
            self.log.add_row(
                "workflow_run",
                f"Results saved to [green]{savef}[/green] for [blue]{data_node.name}[/blue]",
            )
        else:
            self.log.add_row(
                "workflow_run", f"All results saved to [green]{savef}[/green]"
            )

    # ==================== Node Execution Methods ====================

    def _run_nodes(
        self,
        nodes: list[TaskNode | DataNode] | list[TaskNode] | list[DataNode],
        *args,
        **kwargs,
    ):
        """Execute a list of nodes (tasks or data nodes) in the workflow.

        This method recursively processes nodes, handling both task execution
        and data node updates. It maintains visualization state and manages
        child node execution.
        """
        single_run = kwargs.get("single", False)
        render = kwargs.get("render", False)

        self._mark_nodes_as_processing(nodes)
        self.render("workflow_run", file=kwargs.get("value", ""), render=render)

        child_tasks = self._execute_nodes(nodes, args, kwargs, render)

        if child_tasks and not single_run:
            self._run_nodes(child_tasks, *args, **kwargs)

    def _mark_nodes_as_processing(
        self, nodes: list[TaskNode | DataNode] | list[TaskNode] | list[DataNode]
    ):
        """Mark nodes as currently being processed in the graph."""
        for node in nodes:
            graph_node = self.flow.nodes[str(node.name)]
            graph_node["color"] = self.COLOR_PROCESSING
            graph_node["fontcolor"] = self.COLOR_FONT

    def _execute_nodes(
        self,
        nodes: list[TaskNode | DataNode] | list[TaskNode] | list[DataNode],
        args: tuple,
        kwargs: dict,
        render: bool,
    ) -> list[TaskNode]:
        """Execute all nodes and collect child tasks.

        Returns:
            List of child tasks that are ready to execute.
        """
        child_tasks_of_data_nodes = dict()

        for node in nodes:
            if isinstance(node, TaskNode):
                self._execute_task_node(node, args, kwargs, render)
            elif isinstance(node, DataNode):
                child_tasks = self._execute_data_node(node, args, kwargs, render)
                child_tasks_of_data_nodes.update(child_tasks)

        return self._check_tasks_ready(list(child_tasks_of_data_nodes.values()))

    def _execute_task_node(
        self, node: TaskNode, args: tuple, kwargs: dict, render: bool
    ):
        """Execute a single task node."""
        if self._check_task_ready(node) != NodeStatus.READY:
            logger.info(f"[yellow]Task {node.name} is not ready, skipping.[/yellow]")
            return

        children = self._get_node_children(node)
        results = self._run_task(node, **kwargs)

        self._update_node_status(node, bool(results))
        self.render("workflow_run", file=kwargs.get("value", ""), render=render)

        self._run_nodes(children, *results, **kwargs)

    def _execute_data_node(
        self, node: DataNode, args: tuple, kwargs: dict, render: bool
    ) -> dict:
        """Execute a single data node and return child tasks.

        Returns:
            Dictionary of child task names to TaskNode objects.
        """
        # Process incoming data
        for arg in args:
            if (
                isinstance(arg, Data)
                and node.name == arg.get_name()
                and node.check_type(type(arg))
            ):
                self._check_existing(node, arg)

        # Save results
        self.save_run(kwargs.get("key", ""), kwargs.get("value", ""), data_node=node)

        # Update visualization
        self._update_node_status(node, True)
        self.render("workflow_run", file=kwargs.get("value", ""), render=render)

        # Collect child tasks
        children = self._get_node_children(node)
        return {child.name: child for child in children if isinstance(child, TaskNode)}

    def _get_node_children(
        self, node: TaskNode | DataNode
    ) -> list[TaskNode | DataNode]:
        """Get all children of a node."""
        children = list(self.flow.successors(str(node.name)))
        return [self.node_manager.resolve_node(child) for child in children]

    def _update_node_status(self, node: TaskNode | DataNode, success: bool):
        """Update node visualization status based on execution result."""
        color = self.COLOR_SUCCESS if success else self.COLOR_FAILURE
        self.flow.nodes[str(node.name)]["color"] = color
        self.flow.nodes[str(node.name)]["fontcolor"] = self.COLOR_FONT

    # ==================== Task Execution Methods ====================

    def _check_task_ready(self, task_node: TaskNode) -> NodeStatus:
        parents = self.node_manager.get_parents_data(task_node)
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

    def _check_existing(self, node: DataNode, new_item: Data) -> None:
        """Check if an identical item already exists in the data node.

        If an identical item is found, update the existing item with the new item's ID.
        If no identical item is found, add the new item to the data node.

        Args:
            node: The data node to check.
            new_item: The new data item to check or add.
        """
        items = node.items
        for existing_id, existing_item in items.items():
            if self._is_identical_item(existing_item, new_item):
                new_item.id = existing_item.id
                items[existing_id] = new_item
                return

        # No identical item found, add as new
        node.add_item(new_item)

    def _is_identical_item(self, original_item: Data, new_item: Data) -> bool:
        """Check if two data items are identical.

        Compares both parent relationships and serialized data attributes.
        """
        if set(new_item.parents) != set(original_item.parents):
            return False

        return self._compare_serialized_data(original_item, new_item)

    def _compare_serialized_data(self, original_item: Data, new_item: Data) -> bool:
        """Compare serialized data of two items, ignoring IDs and parents."""
        temp_original = self._prepare_for_comparison(copy.deepcopy(original_item))
        temp_new = self._prepare_for_comparison(copy.deepcopy(new_item))

        return json.dumps(temp_original, sort_keys=True) == json.dumps(
            temp_new, sort_keys=True
        )

    def _prepare_for_comparison(self, item: Data) -> dict:
        """Prepare data item for comparison by normalizing and removing metadata."""
        serialized = item.serialize()
        serialized.pop("parents", None)
        serialized.pop("id", None)

        # Normalize string data for comparison
        if isinstance(serialized.get("data"), str):
            serialized["data"] = serialized["data"].strip().lower()

        return serialized

    def _run_task(self, task_node: TaskNode, **kwargs) -> list:
        task = task_node.get_task()
        if not isinstance(task, Task):
            raise TypeError(f"Node {task_node.name} is not a Task")
        self.log.add_row(
            "workflow_run", f"Attempting to run task: [blue]{task.name}[/blue]"
        )
        parents = self.node_manager.get_parents_data(task_node)
        entries = task.prepare_inputs(parents)
        results = list()
        if len(entries) == 0:
            self.log.add_row(
                "workflow_run", f"No input data found for task {task.name}, skipping."
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
        output: Data | list[Data] | None = task.run(*entries)
        try:
            if output is None:
                return list()
            if not isinstance(output, list):
                output = [output]
            for result in output:
                result.set_parents(entries)
            return output
        except Exception as e:
            self.log.add_row(
                "workflow_run",
                f"Error occurred while executing entries for task {task.name}: {str(e)}",
                traceback.print_exc(),
                str(entries),
            )
            return list()

    def train(self, task_name: str, **kwargs) -> tuple:
        """Train a machine learning task.

        Args:
            task_name: Name of the ML task to train.
            **kwargs: Training parameters including 'input' paths.

        Returns:
            Tuple of (processed_inputs, validation_results).
        """
        self._ensure_workflow_connected()
        task = self._get_ml_task(task_name)
        children = self.node_manager.get_children_data(
            self.node_manager.resolve_node(task_name)
        )

        processed_inputs = self._prepare_training_data(kwargs, children)
        kwargs = self._merge_task_params(task_name, kwargs)

        return self._execute_training(task, processed_inputs, kwargs)

    def _get_ml_task(self, task_name: str) -> MachineLearningTask:
        """Get and validate machine learning task."""
        task_node = self.node_manager.resolve_node(task_name)
        if not isinstance(task_node, TaskNode):
            raise TypeError(f"Task {task_name} not found in workflow")

        task = task_node.get_task()
        if not isinstance(task, MachineLearningTask):
            raise TypeError(f"Task {task_name} is not a MachineLearningTask")

        return task

    def _prepare_training_data(self, kwargs: dict, children: list[DataNode]) -> list:
        """Prepare training data from input paths."""
        processed_inputs = []
        inputs = kwargs.pop("input", [])

        kwargs["classes"] = kwargs.get("classes", [child.name for child in children])

        for input_path in inputs:
            item_files = self._get_training_files(Path(input_path))
            for item_file in item_files:
                processed_inputs.extend(DataNode.batch_process(item_file, children))

        return processed_inputs

    def _get_training_files(self, input_path: Path) -> list[Path]:
        """Get training files from input path."""
        if input_path.is_dir():
            return list(input_path.glob(f"*.{self.ext}"))
        elif input_path.suffix == f".{self.ext}":
            return [input_path]
        return []

    def _merge_task_params(self, task_name: str, kwargs: dict) -> dict:
        """Merge task-specific parameters with provided kwargs."""
        task_params = self.data.get(task_name, dict()).get("params", dict())
        return kwargs | task_params

    def _execute_training(
        self, task: MachineLearningTask, processed_inputs: list, kwargs: dict
    ) -> tuple:
        """Execute the training process and save results."""
        with change_dir(self.path):
            results, validation_results = task.train(*processed_inputs, **kwargs)
            self.save_train(task, results)
        return results, validation_results

    def save_train(self, task: MachineLearningTask, results: DetMetrics):
        """Save trained model and update configuration.

        Args:
            task: The machine learning task that was trained.
            results: Training results containing model path.
        """
        best_model_path = results.save_dir / "weights" / "best.pt"

        if not best_model_path.exists():
            raise Exception(
                f"[red]Best model file not found at {best_model_path}[/red]"
            )

        logger.info(f"Best model found at: [green]{best_model_path}[/green]")

        # Check if we need to update the model path
        task_data = self.data.get(task.name, dict())
        current_model = task_data.get("model", "")

        if best_model_path.name != current_model:
            new_model_path = self._generate_model_filename(task.name)
            logger.info(f"Updating model for task {task.name} to {new_model_path}")
            shutil.copy(best_model_path, new_model_path)
            self.data[task.name]["model"] = new_model_path

    def _generate_model_filename(self, task_name: str) -> str:
        """Generate a timestamped model filename."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{task_name}.pt"

    def ensemble(self, folder: list[Path], output: str | Path):
        """Create ensembled files from source folders.

        Delegates to EnsembleProcessor for the actual ensemble logic.

        Args:
            folder: List of source folders containing collectra results
            output: Output folder path for ensembled results
        """
        from collectra.pipelines.ensemble import EnsembleProcessor

        self._ensure_workflow_connected()
        processor = EnsembleProcessor(
            node_manager=self.node_manager,
            flow=self.flow,
            ext=self.ext,
            name=self.name,
            version=self.version,
        )
        processor.ensemble(folder, Path(output))

    @property
    def metadata(self) -> dict:
        """Get pipeline metadata dictionary."""
        return {
            self.PIPELINE_METADATA_KEY: {
                "name": self.name,
                "ext": self.ext,
                "version": self.version,
            }
        }

    def save(self):
        """Save the workflow configuration to pipeline.yaml."""
        with change_dir(self.path):
            config = self.metadata | self.data
            self._write_pipeline_config(config)

    def _write_pipeline_config(self, config: dict):
        """Write pipeline configuration to YAML file."""
        with open("pipeline.yaml", "w") as f:
            for key, value in config.items():
                yaml.dump({key: value}, f, sort_keys=False)
                f.write("\n")

    # ==================== Utility and Helper Methods ====================

    def _get_io_list(self, io: list | str) -> list:
        return [io] if isinstance(io, str) else io

    def _check_task_io(self, io_type: str, task: Task, task_config: dict) -> list:
        """Extract IO node information from task configuration.

        Args:
            io_type: Either 'input' or 'output'.
            task: The task object.
            task_config: Configuration dictionary for the task.

        Returns:
            List of tuples containing (node_name, types).
        """
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
        """Check and resolve data references in configuration.

        If a value references another key in collectra_data, replace it with
        the actual value.

        Args:
            collectra_data: The main workflow data dictionary.
            data: The dictionary to check for references.

        Returns:
            Updated data dictionary with resolved references.
        """
        for key, value in data.items():
            if isinstance(value, str) and value in collectra_data:
                data[key] = collectra_data[value]
        return data

    # ==================== Workflow Graph Construction ====================

    def connect(self):
        """Initialise the DAG representing the workflow. Preparing it for execution."""
        relations: dict[str, dict] = dict()
        with change_dir(self.path):
            self._build_nodes_and_relations(relations)
            self._create_graph_edges(relations)

    def _build_nodes_and_relations(self, relations: dict):
        """Build workflow nodes and track their relationships."""
        for name, value in self.data.items():
            data = copy.deepcopy(value)
            if not isinstance(data, dict) or "type" not in data:
                continue

            cls_ = load_class_from_string(data.pop("type"))
            data = self._check_data_assignment(self.data, data)
            obj = self._create_or_reuse_object(name, cls_, data)

            if isinstance(obj, Task):
                self._process_task_node(obj, data, relations)
            else:
                self.node_manager.add_data_node(name, obj=obj)

    def _create_or_reuse_object(self, name: str, cls_: type, data: dict):
        """Create new object or reuse existing task from graph."""
        if issubclass(cls_, Task) and self.flow.nodes.get(name, None):
            return self.flow.nodes[name]["node"].get_task()
        return cls_(name, **data)

    def _process_task_node(self, task: Task, data: dict, relations: dict):
        """Process a task node by adding it to the graph and tracking IO."""
        task_key = task.name
        relations[task_key] = {
            "input": self._get_io_list(data.get("input", [])),
            "output": self._get_io_list(data.get("output", [])),
        }
        self.node_manager.add_task_node(task_key, task)

        # Add data nodes for task inputs and outputs
        ios = self._extract_task_io_nodes(task, data)
        for io_key, io_types in ios:
            self.node_manager.add_data_node(io_key, types=list(io_types))

    def _extract_task_io_nodes(
        self, task: Task, task_config: dict
    ) -> list[tuple[str, tuple]]:
        """Extract input and output node information from task configuration."""
        ios = []
        ios.extend(self._check_task_io("input", task, task_config))
        ios.extend(self._check_task_io("output", task, task_config))
        return ios

    def _create_graph_edges(self, relations: dict):
        """Create edges in the workflow graph based on task relationships."""
        for task_name, io in relations.items():
            inputs = io.get("input", [])
            outputs = io.get("output", [])

            for input_node in inputs:
                if self.flow.has_node(input_node):
                    self.flow.add_edge(input_node, task_name)

            for output_node in self._get_io_list(outputs):
                if self.flow.has_node(output_node):
                    self.flow.add_edge(task_name, output_node)

    # ==================== Visualization Methods ====================

    def render(self, dest: str | Path = "", file="", render=False) -> None:
        """Render the workflow graph as an SVG visualization.

        Args:
            dest: Destination path for the rendered file.
            file: Optional file being processed (shown in title).
            render: Whether to actually render (allows conditional rendering).
        """
        if not render:
            return

        self._ensure_workflow_connected()
        visual_graph = self._prepare_graph_for_visualization()
        dot_str = self._generate_dot_string(visual_graph, file)

        dest_path = Path(dest) if dest else self.path / "workflow"
        graphviz.Source(dot_str).render(filename=dest_path, format="svg", cleanup=True)
        time.sleep(0.1)  # Brief pause for file system

    def _prepare_graph_for_visualization(self) -> nx.DiGraph:
        """Prepare a copy of the graph for visualization.

        Removes internal node objects that shouldn't be visualized.
        """
        visual_graph = self.flow.copy()
        for node_id in visual_graph.nodes:
            node_data = visual_graph.nodes[node_id]
            if "node" in node_data:
                del node_data["node"]
        return visual_graph

    def _generate_dot_string(self, graph: nx.DiGraph, file: str = "") -> str:
        """Generate DOT format string from graph.

        Args:
            graph: The networkx graph to convert.
            file: Optional file name to include in graph title.

        Returns:
            DOT format string representation of the graph.
        """
        dot_str = nx.nx_pydot.to_pydot(graph).to_string()

        if file:
            title = f"{self.name} Workflow - Processing: {file}"
            dot_str = dot_str.replace(
                "{", f'{{\nlabel="{title}";\nlabelloc="t";\nfontsize=20;\n\n', 1
            )

        return dot_str
