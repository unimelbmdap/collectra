"""Node graph management for Collectra workflows.

This module provides the NodeGraphManager class for handling node graph operations
including node resolution, parent/child data retrieval, and node addition.
"""

from typing import List, Set, Union

import networkx as nx

from ..commons.base import NodeStatus
from ..tasks.base import Task, TaskNode
from ..types.base import Artefact, ArtefactNode, Node


class NodeGraphManager:
    """Manages node graph operations for the Collectra workflow.

    This class encapsulates all graph manipulation operations including:
    - Node resolution and retrieval
    - Parent and child node queries
    - Adding task and artefact nodes to the graph

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

    def resolve_node(self, node_name: str) -> Union[TaskNode, ArtefactNode]:
        """Get a node from the workflow graph by name.

        Args:
            node_name (str): Name of the node to retrieve.

        Returns:
            Union[TaskNode, ArtefactNode]: The resolved node object.

        Raises:
            ValueError: If the node is not found or is empty.
        """
        node: Union[dict, None] = self.flow.nodes.get(node_name, None)
        if not node:
            raise ValueError(f"{node_name} not found in workflow")
        data: Union[TaskNode, ArtefactNode, None] = node.get("node", None)
        if not data:
            raise ValueError(
                f"data for {node_name} not found in workflow. Possible empty node."
            )
        return data

    # =========================================================================
    # Node Retrieval (hierarchical: get_nodes -> get_task_nodes / get_artefact_nodes)
    # =========================================================================

    def get_nodes(self) -> list[Union[TaskNode, ArtefactNode]]:
        """Return all node objects in the graph."""
        return [node["node"] for node in self.flow.nodes.values()]

    def get_task_nodes(self) -> list[TaskNode]:
        """Return only TaskNode instances from the graph."""
        return [n for n in self.get_nodes() if isinstance(n, TaskNode)]

    def get_artefact_nodes(self) -> list[ArtefactNode]:
        """Return only ArtefactNode instances from the graph."""
        return [n for n in self.get_nodes() if isinstance(n, ArtefactNode)]

    def get_node_names(self) -> list[str]:
        """Return all node names in the graph."""
        return list(self.flow.nodes.keys())

    # =========================================================================
    # Parent Retrieval (hierarchical: get_parents -> get_parents_artefact / get_parents_task)
    # =========================================================================

    def get_parents(self, node: Node) -> list[Union[TaskNode, ArtefactNode]]:
        """Get all resolved parent nodes for a given node."""
        parent_names = list(self.flow.predecessors(str(node.name)))
        return [self.resolve_node(name) for name in parent_names]

    def get_parents_artefact(self, node: Node) -> List[ArtefactNode]:
        """Get all parent artefact nodes for a given node."""
        return [p for p in self.get_parents(node) if isinstance(p, ArtefactNode)]

    def get_parents_task(self, node: Node) -> list[TaskNode]:
        """Get all parent task nodes for a given node."""
        return [p for p in self.get_parents(node) if isinstance(p, TaskNode)]

    def get_ancestor_artefacts(self, node: Node) -> list[ArtefactNode]:
        """Return every artefact node upstream of ``node``."""
        return [
            resolved
            for name in nx.ancestors(self.flow, str(node.name))
            if isinstance((resolved := self.resolve_node(name)), ArtefactNode)
        ]

    # =========================================================================
    # Children Retrieval (hierarchical: get_children -> get_children_artefact / get_children_task)
    # =========================================================================

    def get_children(self, node: Node) -> list[Union[TaskNode, ArtefactNode]]:
        """Get all resolved child nodes for a given node."""
        child_names = list(self.flow.successors(str(node.name)))
        return [self.resolve_node(name) for name in child_names]

    def get_children_artefact(self, node: Node) -> List[ArtefactNode]:
        """Get all child artefact nodes for a given node."""
        return [c for c in self.get_children(node) if isinstance(c, ArtefactNode)]

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
            if isinstance(node, ArtefactNode):
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

    def add_artefact_node(
        self,
        name: str,
        obj: Union[Artefact, None] = None,
        types: Union[List[type], Set[type]] = [],
    ) -> None:
        """Add an artefact node to the workflow graph.

        If the node already exists, updates it with the new object and types.
        Otherwise, creates a new ArtefactNode and adds it to the graph.

        Args:
            name (str): Name of the artefact node.
            obj (Union[Artefact, None], optional): The artefact object to store. Defaults to None.
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
            node = ArtefactNode(name, items=items, types=types)
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
            artefact_node: ArtefactNode = node["node"]
            if not isinstance(artefact_node, ArtefactNode):
                return
            if type(obj) not in artefact_node.types:
                return
            artefact_node.add_item(obj)
            artefact_node.types = artefact_node.types.union(set(types))
