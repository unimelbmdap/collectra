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
from rich import print
from ultralytics.utils.metrics import DetMetrics


from collectra import Node, NodeStatus, Task, TaskNode, Data, DataNode, MachineLearningTask
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
        """Runs the workflow from the specified task or from all root tasks.        
        """        
        if self.flow.number_of_nodes() == 0:
            self.connect()
        roots: list = list()
        if task_name:
            task_node = self._resolve_node(task_name)
            assert isinstance(task_node, TaskNode), f"Task {task_name} not found in workflow"
            roots.append(task_node)
        else:
            task_nodes = [node["node"] for node in self.flow.nodes.values() if isinstance(node["node"], TaskNode)]
            for task_node in task_nodes:                
                is_root = self._check_is_root(task_node._name)                                    
                if is_root:                 
                    roots.append(task_node)     

        single = kwargs.pop("single", False)        
        output = kwargs.pop("output", None)        
        for key, value in kwargs.items():                  
            input = {key: value}            
            self._run_nodes(roots, single=single, **input)                  
            self.save_run(key, value)                        

    def save_run(self, key: str, value: str | Path):           
        savef = Path(value)
        data_nodes = [node["node"] for node in self.flow.nodes.values() if isinstance(node["node"], DataNode)]                              
        if savef.is_file():
            savef = savef.parent / Path(savef.name.replace(savef.suffix, f".{self.ext}"))            
            savef.mkdir(parents=True, exist_ok=True)
        if savef.is_dir():                        
            with change_dir(savef):            
                results = dict()    
                resultsf = Path("results.yaml")                
                if resultsf.exists():
                    with open(resultsf, "r") as f:
                        results = yaml.safe_load(f)
                    results["collectra_results_metadata"]["timestamp"] = datetime.datetime.now().isoformat()
                else:
                    results["collectra_results_metadata"] = {
                        "workflow": self.name,
                        "version": self.version,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }                                    
                
                for data_node in data_nodes:
                    if data_node.status != NodeStatus.READY:
                        continue                
                    name = data_node.name
                    if name in results:
                        results[name] = None
                    for item in data_node.items:
                        item_data = item.serialize()                        
                        if name not in results or results[name] is None:
                            results[name] = list()                         
                        results[name].append(item_data)
                    if len(results[name]) == 1:
                        results[name] = results[name][0]

                with open("results.yaml", "w") as f:
                    for key, data in results.items():
                        yaml.dump({key: data}, f, sort_keys=False)
                        f.write("\n")                      

            if Path(value).is_file():
                shutil.copy(Path(value), savef / Path(value).name)    
        
        print(f"Results saved to [green]{savef}[/green]")
        
    def _resolve_node(self, node_name: str) -> TaskNode | DataNode:        
        node: dict | None = self.flow.nodes.get(node_name, None)
        assert node, f"{node_name} not found in workflow"
        data: TaskNode | DataNode | None = node.get("node", None)
        assert data, f"data for {node_name} not found in workflow. Possible empty node."
        return data
    
    def _get_parents_data(self, node: Node) -> list[DataNode]:
        parents: list[DataNode] = list()
        parent_names = list(self.flow.predecessors(str(node._name)))        
        for parent_name in parent_names:
            parent_node = self._resolve_node(parent_name)            
            if isinstance(parent_node, DataNode):
                parents.append(parent_node)
        return parents

    def _get_children_data(self, node: Node) -> list[DataNode]:
        return self._get_data_nodes(list(self.flow.successors(str(node._name))))

    def _get_data_nodes(self, node_names: list[str]) -> list[DataNode]:
        nodes: list[DataNode] = list()
        for node_name in node_names:
            node = self._resolve_node(node_name)
            if isinstance(node, DataNode):
                nodes.append(node)
        return nodes

    def _run_nodes(self, nodes: list[TaskNode | DataNode] | list[TaskNode] | list[DataNode], *args, **kwargs): 
        single_run = kwargs.get("single", False)    
        children_tasks_of_data: list[TaskNode] = list()                      
        for node in nodes:                                              
            if isinstance(node, TaskNode):
                parents = self._get_parents_data(node)                
                results: list = self._run_task(node, parents=parents, **kwargs)                                                                                  
                children = list(self.flow.successors(str(node._name)))
                children = [self._resolve_node(child) for child in children]      
                if single_run:                                                
                    children = [child for child in children if isinstance(child, DataNode)]                
                self._run_nodes(children, *results, **kwargs)                            
            elif isinstance(node, DataNode):                                                  
                for arg in args:  
                    if not isinstance(arg, Data):
                        continue
                    if node.name == arg.get_name() and node.check_type(type(arg)):                    
                        node.add_item(arg)    
                children = list(self.flow.successors(str(node._name)))
                children = [self._resolve_node(child) for child in children]                              
                for child in children:
                    if isinstance(child, TaskNode):
                        exists = False
                        for existing in children_tasks_of_data:
                            if existing.name ==  child.name:
                                exists = True
                                break        
                        if not exists:
                            children_tasks_of_data.append(child)               
        if len(children_tasks_of_data) > 0:
            self._run_nodes(children_tasks_of_data, *args, **kwargs)

    def _run_task(self, task_node: TaskNode, parents: list[DataNode], **kwargs) -> list:        
        task = task_node.get_task()        
        print(f"running task: [blue]{task.name}[/blue]")
        assert isinstance(task, Task), f"Node {task_node.name} is not a Task"        
        entries: list = list()                         
        for parent in parents:                        
            value = kwargs.get(parent.name, None)            
            if parent.status != NodeStatus.READY:                
                parent.process(parent.name, value, **kwargs)                                
            entries.extend(parent._items)             
        results: list = list()                              
        with change_dir(self.path):                                                      
            for index in range(0, len(entries), task.input_nums):
                endindex = len(entries) if index + task.input_nums > len(entries) else index + task.input_nums                
                sub_entries = entries[index : endindex]                
                result = task.run(*sub_entries)                    
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)                                                                 
        return results

    def train(self, task_name: str, **kwargs) -> tuple:
        if self.flow.number_of_nodes() == 0:
            self.connect()                
        task_node = self._resolve_node(task_name)      
        assert isinstance(task_node, TaskNode), f"Task {task_name} not found in workflow"
        task = task_node.get_task()
        assert isinstance(task, MachineLearningTask), f"Task {task_name} is not a MachineLearningTask"                
        children = self._get_children_data(task_node)         
        processed_inputs: list = list()                 
        inputs = kwargs.pop("input", [])            
        for input in inputs:                                     
            input_path = Path(input) 
            item_files = list(input_path.glob(f"*.{self.ext}")) if input_path.is_dir() else [input_path] if input_path.suffix == self.ext else [] 
            for item_file in item_files:                
                with change_dir(item_file):                                          
                    processed_inputs.extend(DataNode.batch_process(children))                                                                                       
        kwargs["classes"] = [child._name for child in children] if not "classes" in kwargs else kwargs["classes"]
        kwargs = kwargs | self.data.get(task_name, dict()).get("params", dict())
        with change_dir(self.path):                                 
            processed_inputs, validation_results = task.train(*processed_inputs, **kwargs)
            self.save_train(task, processed_inputs)    
        return processed_inputs, validation_results
    
    def save_train(self, task: MachineLearningTask, results: DetMetrics):        
        new_model_path = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{task.name}.pt"  
        best_model_path = results.save_dir / "weights" / "best.pt"              
        task_data = self.data.get(task.name, dict())
        if best_model_path.name != task_data["model"]:
            shutil.copy(best_model_path, new_model_path)
            if Path(task_data["model"]).exists():
                os.remove(task_data["model"])
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
        param_types = list(unpack_types(task.run, get_param_types if io_type == "input" else get_return_type).items())        
        io = self._get_io_list(task_config.get(io_type, []))        
        nodes: list = list()
        for item in io:
            for _, types in param_types:                 
                if not isinstance(types, tuple):
                    types = (types,)
                for type_ in types:                                     
                    nodes.append((item, type_))
        return nodes                 

    def connect(self):
        """Initialise the DAG representing the workflow. Preparing it for execution.""" 
        relations: dict[str, dict] = dict()                    
        with change_dir(self.path):            
            for name, value in self.data.items():                
                data = copy.deepcopy(value)
                cls_ = load_class_from_string(data.pop("type"))
                obj = cls_(name, **data)                
                if isinstance(obj, Task):
                    task_key = obj.name
                    relations[task_key] = {
                        "input": self._get_io_list(data.get("input", [])),
                        "output": self._get_io_list(data.get("output", [])),
                    } 
                    self._add_task_node(task_key, obj)   
                    ios: list[tuple[str, type]] = list()                                                                                                                      
                    ios.extend(self._check_task_io("input", obj, data))                    
                    ios.extend(self._check_task_io("output", obj, data))                    
                    for io_key, io_type in ios:
                        self._add_data_node(io_key, type_=io_type)                                                                                                                                    
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
    
    def _add_data_node(self, name, obj: Data | None = None, type_: type | None = None):
        node = self.flow.nodes.get(name, None)
        type_ = type(obj) if obj else type_
        if not node:
            items = [obj] if obj else []
            types = set([type_]) if type_ else set()
            node = DataNode(name, _items=items, _types=types)
            self.flow.add_node(
                name,
                node=node,
                label=str(node),
                shape="oval",
                color="salmon",
                fontcolor="black",
                style="filled",
            )
        else:                             
            data = node["node"]            
            if type_:
                data.add_type(type_)   
            if obj:
                data.add_item(obj)

    def render(self, dest: str | Path = ""):
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
        graphviz.Source(dot_str).render(filename=dest, format="svg", cleanup=True)
