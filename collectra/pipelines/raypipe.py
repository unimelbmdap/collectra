"""Ray-based distributed workflow execution for Collectra (experimental).

This module provides an experimental implementation of Collectra workflows
using Ray for distributed and parallel execution. It's designed to handle
large-scale data processing workflows that can benefit from distributed
computing capabilities.

Note: This is an experimental module and may not have full feature parity
with the main Collectra pipeline implementation.

Classes:
    Collectra: Ray-enabled workflow class for distributed execution
"""

from pathlib import Path
from dataclasses import dataclass, field
from collectra.tasks.base import Task
import networkx as nx
import ray

@dataclass(kw_only=True)
class Collectra:
    name: str
    version: str
    format: str
    dest: str | Path | None = None
    mode: str = "file"
    description: str = "Collectra workflow configuration"
    tasks: list[Task] = []
    flow: nx.DiGraph | None = None

    def __post_init__(self):
        if not self.dest:
            self.dest = str(Path.cwd() / self.name)
        self.flow = nx.DiGraph()

    def add_task(self, task: Task, mapping: dict):
        
        if self.flow is None:
            raise ValueError(f"Flow has not been initialised for this pipeline")
        if task.name in self.flow.nodes():
            raise ValueError(f"Flow already has this task: {task.name}")
        self.tasks.append(task)                
        self.flow.add_node(task.name, item=task)        
        for output in task.output:
            if output not in mapping:
                mapping[output] = []
            mapping[output].append(task.name)        
            
    def connect(self, mapping: dict):   
        if self.flow is None:
            raise ValueError(f"Flow has not been initialised for this pipeline")     
        for task in self.tasks:
            for input in task.input:                
                parent_tasks = mapping.get(input, [])
                for parent_task in parent_tasks:
                    self.flow.add_edge(parent_task, task.name)
        