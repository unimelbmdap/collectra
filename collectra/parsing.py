"""Workflow parsing and graph generation for Collectra workflows.

This module provides classes and functions for parsing YAML workflow configurations
and converting them into directed graphs using NetworkX. It defines the basic
building blocks for workflow nodes and specialized task types.

The module supports:
    - Loading YAML workflow configurations
    - Creating directed graphs from workflow definitions
    - Specialized task types (TextProcessor, ObjectDetection, OCR)
    - Graph visualization and export functionality

Classes:
    CollectraNode: Base class for workflow nodes
    CollectraTask: Base class for executable tasks
    TextProcessor: Task for text processing operations
    ObjectDetection: Task for object detection operations
    OCR: Task for optical character recognition
    CollectraWorkflow: Main workflow class for parsing and execution
"""

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import yaml

from collectra.utils import from_dir, unzip


@dataclass(kw_only=True)
class CollectraNode:
    """Base class for all nodes in a Collectra workflow graph.

    Represents a single node in the workflow directed graph with input and output
    connections. Serves as the foundation for more specialized node types.

    Attributes:
        name (str): Unique identifier for the node within the workflow.
        input (list[str]): List of input connection names for this node.
        output (list[str]): List of output connection names from this node.
    """

    name: str
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Convert string inputs/outputs to lists for consistency."""
        if isinstance(self.input, str):
            self.input = [self.input]
        if isinstance(self.output, str):
            self.output = [self.output]

    def set_node_attributes(self, node):
        """Set visual attributes for the node in graph rendering.

        Args:
            node: NetworkX node object to set attributes on.
        """
        pass


@dataclass(kw_only=True)
class CollectraTask(CollectraNode):
    """Base class for executable tasks in a Collectra workflow.

    Extends CollectraNode with task-specific functionality and visual styling
    for graph rendering. Tasks represent executable operations in the workflow.
    """

    def set_node_attributes(self, node):
        """Set visual attributes for task nodes in graph rendering.

        Args:
            node: NetworkX node object to set attributes on.
        """
        node["color"] = "dodgerblue"
        node["shape"] = "box"


@dataclass(kw_only=True)
class TextProcessor(CollectraTask):
    """Task for text processing operations in workflows.

    Specialized task type for handling text analysis, transformation,
    and natural language processing operations.
    """

    pass


@dataclass(kw_only=True)
class ObjectDetection(CollectraTask):
    """Task for object detection operations in workflows.

    Specialized task type for computer vision operations that identify
    and locate objects within images or video streams.
    """

    pass


@dataclass(kw_only=True)
class OCR(CollectraTask):
    """Task for optical character recognition operations in workflows.

    Specialized task type for extracting text content from images
    and converting it to machine-readable text format.
    """

    pass


def load_class_from_string(path: str | None):
    """Dynamically load a class from a string module path.

    Takes a fully qualified class path string and imports the class
    for instantiation. Used for loading task classes from configuration.

    Args:
        path (str): Fully qualified class path (e.g., 'module.submodule.ClassName').

    Returns:
        type: The loaded class object ready for instantiation.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


@dataclass()
class CollectraWorkflow:
    """Main workflow class for parsing and executing Collectra workflows.

    Loads workflow configurations from YAML files and converts them into
    executable directed graphs. Manages the complete workflow lifecycle
    including parsing, validation, and execution.

    Attributes:
        path (Path): Path to the workflow configuration YAML file.
        dag (nx.DiGraph): Directed graph representation of the workflow.
    """

    path: Path
    dag: nx.DiGraph = field(init=False, default=None)

    def __post_init__(self):
        """Initialize the workflow by reading the YAML configuration."""
        self.read_yaml()

    def read_yaml(self) -> nx.DiGraph:
        """Parse the YAML workflow configuration into a directed graph.

        Reads the workflow definition from the specified YAML file and constructs
        a NetworkX directed graph with nodes representing tasks and edges
        representing data flow dependencies.

        Returns:
            nx.DiGraph: The constructed workflow graph.

        Raises:
            AssertionError: If required 'type' field is missing from any task.
            FileNotFoundError: If the workflow YAML file cannot be found.
            yaml.YAMLError: If the YAML file is malformed.
        """
        path = Path(self.path)
        with path.open("r") as f:
            data = yaml.safe_load(f)

        self.dag = nx.DiGraph()

        items = dict()

        for name, kwargs in data.items():
            type_ = kwargs.pop("type", None)
            assert type_ is not None, f"Type is required for {name}"

            cls = load_class_from_string(type_)
            item = cls(name=name, **kwargs)
            assert isinstance(item, CollectraNode)
            items[name] = item

            if name not in self.dag:
                self.dag.add_node(name, item=item)

            node = self.dag.nodes[name]
            node["item"] = item

            # Set colour and attributes of node in networkx
            item.set_node_attributes(node)

            for input_name in item.input:
                self.dag.add_edge(input_name, name)

            for output_name in item.output:
                self.dag.add_edge(name, output_name)

        return self.dag

    def dot(self) -> str:
        """Convert the workflow graph to DOT format string.

        Generates a DOT (Graphviz) representation of the workflow graph
        that can be used for visualization and rendering.

        Returns:
            str: DOT format string representation of the workflow graph.
        """
        return nx.nx_pydot.to_pydot(self.dag).to_string()

    def render(self, output: Path | str) -> str:
        """Render the workflow graph to a visual format.

        Creates a visual representation of the workflow graph and saves it
        to the specified output path. Supports various formats based on
        the file extension (SVG, PNG, PDF, etc.).

        Args:
            output (Path | str): Output file path for the rendered graph.

        Returns:
            str: DOT format string used for rendering.

        Raises:
            ImportError: If graphviz package is not available for rendering.
            OSError: If the output directory cannot be created or written to.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        dot_string = self.dot()
        suffix = output.suffix.lower()
        if suffix == ".dot":
            with output.open("w") as f:
                f.write(dot_string)
        else:
            import graphviz

            graph = graphviz.Source(dot_string)
            format = suffix[1:] if suffix else "svg"
            graph.render(str(output.with_suffix("")), format=format, cleanup=True)

        return dot_string


def parse_item(data):
    if isinstance(data, dict) and "type" in data:
        type_name = data.pop("type")
        type = load_class_from_string(type_name)
        data = type(**data)
    return data


def parse_results(file_path: Path | str) -> dict:
    file_path = Path(file_path).absolute()

    cwd = Path.cwd()

    # set current working directory to 'file'
    if file_path.is_dir():
        os.chdir(file_path)

    results = (
        unzip(file_path, "results.yaml")
        if file_path.is_file()
        else from_dir(file_path, "results.yaml") if file_path.is_dir() else dict()
    )
    for key, data in results.items():
        if isinstance(data, list):
            results[key] = [parse_item(data_item) for data_item in data]
        else:
            results[key] = parse_item(data)

    # Revert current working directory
    os.chdir(cwd)

    return results
