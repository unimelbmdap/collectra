from collectra.tasks.base import Task
from collectra.tasks.ml import MachineLearningTask
from collectra.tasks.managers import TaskManager
from collectra.utils import error_msg, success_msg, processing_msg
from dataclasses import dataclass, field
from pathlib import Path
from rich import print
import networkx as nx, graphviz
import traceback

@dataclass(kw_only=True)
class Collectra:
    name: str
    version: str
    format: str
    out_dir: str | None = None
    as_dir: bool = False
    description: str = "Collectra workflow configuration"
    tasks: list[Task] = field(default_factory=list)      
    flow: nx.DiGraph | None = None
    results: dict = field(default_factory=dict)

    def __post_init__(self):       
        if not self.out_dir:             
            self.out_dir = str(Path.cwd() / self.name)
        self.flow = nx.DiGraph()

    def add_task(self, task: Task, mapping: dict):
        if self.flow is None:
            raise ValueError(f"Pipeline has not been initialised for {self.name}")
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
            raise ValueError(f"Pipeline has not been initialised for {self.name}")  
        for task in self.tasks:
            for input in task.input:                
                parent_tasks = mapping.get(input, [])
                for parent_task in parent_tasks:
                    self.flow.add_edge(parent_task, task.name)     

    def render(self, filename: str = "", raw=False):
        dot_str = ""
        filename = filename if filename else f"{self.name}_DAG_raw" if raw else f"{self.name}_DAG" 
        if raw:
            if self.flow is None:
                raise ValueError(f"Pipeline has not been initialised for workflow: {self.name}")
            dot_str = nx.nx_pydot.to_pydot(self.flow).to_string()
            print(dot_str)
        else:            
            dag = nx.DiGraph()
            if not self.tasks:
                print(error_msg("Skipping rendering empty workflow..."))
                return
            for task in self.tasks:        
                task_name = f"{task.name}\n{task.config['type'].split('.')[-1]}"    
                if task.name not in dag:
                    dag.add_node(task_name, item=task, shape="box", style="filled", fillcolor="blue", fontcolor="white")                                                                
                for input_name in task.input:
                    dag.add_node(input_name, shape="oval", style="filled", fillcolor="green", fontcolor="white")
                    dag.add_edge(input_name, task_name)
                for output_name in task.output:
                    dag.add_node(output_name, shape="oval", style="filled", fillcolor="orange")
                    dag.add_edge(task_name, output_name)                            
            dot_str = nx.nx_pydot.to_pydot(dag).to_string()            
        if self.as_dir and self.out_dir:
            graphviz.Source(dot_str).render(
                filename=f"{Path(self.out_dir) / filename}", format="svg", cleanup=True
            )
        else:
            graphviz.Source(dot_str).render(filename=filename, format="svg", cleanup=True)        
        print(success_msg(f"Workflow rendered to {filename}.svg"))

    def __str__(self) -> str:
        return f"{self.name} v{self.version}"

    def get_config(self, config: dict = dict()) -> dict:
        """
        Generate a YAML representation of the workflow configuration.

        This method serializes the workflow configuration into a YAML format.
        """
        config["collectra_pipeline_metadata"] = {
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "description": self.description,
        }

        for task in self.tasks:
            task_key: str = str(task.name)
            config[task_key] = {
                "type": f"{task.__class__.__module__}.{task.__class__.__name__}"
            }            
            if task.config.get("model", None):
                config[task_key]["model"] = task.config.get("model")  # type: ignore
            if task.input:
                config[task_key]["input"] = task.input if len(task.input) > 1 else task.input[0]  # type: ignore
            if task.output:
                config[task_key]["output"] = task.output if len(task.output) > 1 else task.output[0]  # type: ignore           
            if task.config.get("params", None): 
                config[task_key]["params"] = task.config.get("params")

        return config

    def metadata(self):
        """
        Retrieve the metadata of the current workflow.

        This method returns the metadata of the workflow as a dictionary.
        """
        return {
            "name": self.name,
            "version": self.version,
            "format": self.format,
        }

    def _get_existing_task_ids(self) -> list[str]:
        """
        Get a list of existing task IDs in the workflow.

        This method returns a list of task IDs that are already present in the workflow.
        """
        return [task.name for task in self.tasks]
    
    def __call__(self, **kwargs): 
        self.run(**kwargs)
    
    def _run_nodes(self, pipeline_result: "CollectraResult",  nodes: list[str], **kwargs):        
        if self.flow is None:
            raise ValueError(f"Pipeline is not initialised for: {self.name}") 
        for node in nodes:
            try:
                output_data: dict = self.run_task(node, **kwargs)            
                children_nodes = list(self.flow.successors(node))
                pipeline_result.successful_tasks[node] = output_data
                if children_nodes:
                    self._run_nodes(pipeline_result, children_nodes, **output_data)                
            except Exception as e:
                pipeline_result.failed_tasks[node] = {
                    "error": type(e).__name__,
                    "message": str(e),                    
                }

    def run(self, **kwargs):              
        if self.flow is None:
            raise ValueError(f"Pipeline is not initialised for: {self.name}") 
        result = CollectraResult()
        initial_nodes = [node for node in self.flow.nodes() if len(list(self.flow.predecessors(node))) == 0]            
        self._run_nodes(result, initial_nodes, **kwargs)      
        if result.failed_tasks:
            result.status = "failed" if not result.successful_tasks else "partial"
        print(result)
    
    def save_run(self, **kwargs):
        print(kwargs)

    def run_task(self, task_name: str, **kwargs) -> dict:                
        task = self._get_task(task_name)                                           
        task_config = {            
            "format": self.format,                        
            "as_dir": self.as_dir,
        }                
        task.set_config(task_config)            
        print(processing_msg(f"Running task: {task.name}"))               
        return task(**kwargs)                               

    def _find_task_by_name(self, task_name: str) -> list[Task]:
        return [task for task in self.tasks if task.name == task_name]

    def _get_task(self, task_name: str):
        matching_tasks = self._find_task_by_name(task_name)
        if len(matching_tasks) == 0:
            raise ValueError(f"Task with name {task_name} not found.")
        if len(matching_tasks) > 1:
            raise ValueError(f"Multiple tasks with name {task_name} found.")
        return matching_tasks[0]    

    def train(self, task_name: str, config: dict = dict()):
        """Train the specified task in the workflow.

        Args:
            task_name (str): The name of the task to be trained

        Raises:
            ValueError: If the task is not a machine learning task.
        """
        task = self._get_task(task_name)
        if not isinstance(task, MachineLearningTask):
            raise ValueError(
                error_msg(f"Task with ID {task_name} is not a machine learning task.")
            )
        print(processing_msg(f"Training task: {task.name}"))        
        task_config = {
            **config,
            "format": self.format,
            "task": task.name,
            "inputs": task.input or [],
            "outputs": task.output or [],
            "as_dir": self.as_dir,
        }                
        task.set_config(task_config)        
        model_path = task.train()
        task.set_model(model_path)        

@dataclass(kw_only=True)
class CollectraResult:
    successful_tasks: dict = field(default_factory=dict)
    failed_tasks: dict = field(default_factory=dict)
    status: str = "success"  # success, partial, failed
