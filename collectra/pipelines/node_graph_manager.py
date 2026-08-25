"""Node graph management for Collectra workflows.

This module provides the NodeGraphManager class for handling node graph operations
including node resolution, parent/child data retrieval, and node addition.
"""

from typing import List, Set, Union

import networkx as nx

from ..commons.base import NodeStatus
from ..tasks.base import Task, TaskNode
from ..types.base import Artefact, DataNode, Node


class NodeGraphManager:
    """Manages node graph operations for the Collectra workflow.

    This class encapsulates all graph manipulation operations including:
    - Node resolution and retrieval
    - Parent and child node queries
    - Adding task and data nodes to the graph

    Attributes:
        flow (nx.DiGraph): The NetworkX directed graph representing the workflow.
    """

    def __init__(self):
        """Initialize the NodeGraphManager with an empty graph."""
        self.flow = nx.DiGraph()

    # =========================================================================
    # Graph Lifecycle
    # =========================================================================

    def is_empty(self) -> bool:
        """Returns True if the graph has no nodes."""
        return len(self.flow.nodes) == 0

    def copy_graph(self) -> nx.DiGraph:
        """Return a copy of the graph."""
        return self.flow.copy()

    # =========================================================================
    # Node Resolution
    # =========================================================================

    def resolve_node(self, node_name: str) -> Union[TaskNode, DataNode]:
        """Get a node from the workflow graph by name.

        Args:
            node_name (str): Name of the node to retrieve.

        Returns:
            Union[TaskNode, DataNode]: The resolved node object.

        Raises:
            ValueError: If the node is not found or is empty.
        """
        node: Union[dict, None] = self.flow.nodes.get(node_name, None)
        if not node:
            raise ValueError(f"{node_name} not found in workflow")
        data: Union[TaskNode, DataNode, None] = node.get("node", None)
        if not data:
            raise ValueError(
                f"data for {node_name} not found in workflow. Possible empty node."
            )
        return data

    # =========================================================================
    # Node Retrieval (hierarchical: get_nodes -> get_task_nodes / get_data_nodes)
    # =========================================================================

    def get_nodes(self) -> list[Union[TaskNode, DataNode]]:
        """Return all node objects in the graph."""
        return [node["node"] for node in self.flow.nodes.values()]

    def get_task_nodes(self) -> list[TaskNode]:
        """Return only TaskNode instances from the graph."""
        return [n for n in self.get_nodes() if isinstance(n, TaskNode)]

    def get_data_nodes(self) -> list[DataNode]:
        """Return only DataNode instances from the graph."""
        return [n for n in self.get_nodes() if isinstance(n, DataNode)]

    def get_node_names(self) -> list[str]:
        """Return all node names in the graph."""
        return list(self.flow.nodes.keys())

    # =========================================================================
    # Parent Retrieval (hierarchical: get_parents -> get_parents_data / get_parents_task)
    # =========================================================================

    def get_parents(self, node: Node) -> list[Union[TaskNode, DataNode]]:
        """Get all resolved parent nodes for a given node."""
        parent_names = list(self.flow.predecessors(str(node.name)))
        return [self.resolve_node(name) for name in parent_names]

    def get_parents_data(self, node: Node) -> List[DataNode]:
        """Get all parent data nodes for a given node."""
        return [p for p in self.get_parents(node) if isinstance(p, DataNode)]

    def get_parents_task(self, node: Node) -> list[TaskNode]:
        """Get all parent task nodes for a given node."""
        return [p for p in self.get_parents(node) if isinstance(p, TaskNode)]

    # =========================================================================
    # Children Retrieval (hierarchical: get_children -> get_children_data / get_children_task)
    # =========================================================================

    def get_children(self, node: Node) -> list[Union[TaskNode, DataNode]]:
        """Get all resolved child nodes for a given node."""
        child_names = list(self.flow.successors(str(node.name)))
        return [self.resolve_node(name) for name in child_names]

    def get_children_data(self, node: Node) -> List[DataNode]:
        """Get all child data nodes for a given node."""
        return [c for c in self.get_children(node) if isinstance(c, DataNode)]

    def get_children_task(self, node: Node) -> list[TaskNode]:
        """Get all child task nodes for a given node."""
        return [c for c in self.get_children(node) if isinstance(c, TaskNode)]

    # =========================================================================
    # Node Attributes
    # =========================================================================

    def set_node_attr(self, node_name: str, **attrs) -> None:
        """Set attributes on a graph node."""
        self.flow.nodes[node_name].update(attrs)

    # =========================================================================
    # Graph Construction
    # =========================================================================

    def has_node(self, node_name: str) -> bool:
        """Check if a node exists in the graph."""
        return self.flow.has_node(node_name)

    def add_edge(self, source: str, target: str) -> None:
        """Add an edge between two nodes."""
        self.flow.add_edge(source, target)

    def try_resolve_existing_task(self, name: str, cls_: type) -> Union[Task, None]:
        """If node exists and is a TaskNode subclass, return its Task; else None."""
        node_data = self.flow.nodes.get(name, None)
        if node_data and issubclass(cls_, Task):
            resolved = node_data.get("node", None)
            if resolved and isinstance(resolved, TaskNode):
                return resolved.get_task()
        return None

    # =========================================================================
    # Reset
    # =========================================================================

    def reset_all_nodes(self) -> None:
        """Reset all nodes to their initial state."""
        for node in self.get_nodes():
            if isinstance(node, DataNode):
                node.items = dict()
                node.ensemble_items = dict()
                node.status = NodeStatus.NOT_READY
            if isinstance(node, TaskNode):
                node.status = NodeStatus.NOT_READY

    # =========================================================================
    # Add Nodes
    # =========================================================================

    def add_task_node(self, name: str, obj: Task) -> None:
        """Add a task node to the workflow graph.

        If the node already exists, updates its color attributes.
        Otherwise, creates a new TaskNode and adds it to the graph.

        Args:
            name (str): Name of the task node.
            obj (Task): The task object to wrap in the node.
        """
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

    def add_data_node(
        self,
        name: str,
        obj: Union[Artefact, None] = None,
        types: Union[List[type], Set[type]] = [],
    ) -> None:
        """Add a data node to the workflow graph.

        If the node already exists, updates it with the new object and types.
        Otherwise, creates a new DataNode and adds it to the graph.

        Args:
            name (str): Name of the data node.
            obj (Union[Artefact, None], optional): The data object to store. Defaults to None.
            types (Union[List[type], Set[type]], optional): Expected data types. Defaults to [].
        """
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
