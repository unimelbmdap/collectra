"""Node graph management for Collectra workflows.

This module provides the NodeGraphManager class for handling node graph operations
including node resolution, parent/child data retrieval, and node addition.
"""

import networkx as nx
from typing import List, Set, Union

from ..tasks.base import Task, TaskNode
from ..types.base import Data, DataNode, Node


class NodeGraphManager:
    """Manages node graph operations for the Collectra workflow.
    
    This class encapsulates all graph manipulation operations including:
    - Node resolution and retrieval
    - Parent and child node queries
    - Adding task and data nodes to the graph
    
    Attributes:
        flow (nx.DiGraph): The NetworkX directed graph representing the workflow.
    """
    
    def __init__(self, flow: nx.DiGraph):
        """Initialize the NodeGraphManager.
        
        Args:
            flow (nx.DiGraph): The workflow graph to manage.
        """
        self.flow = flow
    
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
            raise ValueError(f"data for {node_name} not found in workflow. Possible empty node.")
        return data
    
    def get_parents_data(self, node: Node) -> List[DataNode]:
        """Get all parent data nodes for a given node.
        
        Args:
            node (Node): The node to get parents for.
            
        Returns:
            List[DataNode]: List of parent data nodes.
        """
        parents: List[DataNode] = list()
        parent_names = list(self.flow.predecessors(str(node.name)))
        for parent_name in parent_names:
            parent_node = self.resolve_node(parent_name)
            if isinstance(parent_node, DataNode):
                parents.append(parent_node)
        return parents
    
    def get_children_data(self, node: Node) -> List[DataNode]:
        """Get all child data nodes for a given node.
        
        Args:
            node (Node): The node to get children for.
            
        Returns:
            List[DataNode]: List of child data nodes.
        """
        return self.get_data_nodes(list(self.flow.successors(str(node.name))))
    
    def get_data_nodes(self, node_names: List[str]) -> List[DataNode]:
        """Filter and return only data nodes from a list of node names.
        
        Args:
            node_names (List[str]): List of node names to filter.
            
        Returns:
            List[DataNode]: List of data nodes matching the names.
        """
        nodes: List[DataNode] = list()
        for node_name in node_names:
            node = self.resolve_node(node_name)
            if isinstance(node, DataNode):
                nodes.append(node)
        return nodes
    
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
        obj: Union[Data, None] = None, 
        types: Union[List[type], Set[type]] = []
    ) -> None:
        """Add a data node to the workflow graph.
        
        If the node already exists, updates it with the new object and types.
        Otherwise, creates a new DataNode and adds it to the graph.
        
        Args:
            name (str): Name of the data node.
            obj (Union[Data, None], optional): The data object to store. Defaults to None.
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
