from collectra.tasks.base import Task
from collectra.tasks.ml import MachineLearningTask
from collectra.tasks.managers import TaskManager
from collectra.utils import error_msg, success_msg, processing_msg
from dataclasses import dataclass, field
from pathlib import Path
from rich import print
import networkx as nx, graphviz

@dataclass(kw_only=True)
class Collectra:
    name: str
    version: str
    file_format: str
    out_dir: str | None = None
    as_dir: bool = False
    description: str = "Collectra workflow configuration"
    tasks: list[Task] = field(default_factory=list)      

    def __post_init__(self):        
        self.out_dir = str(Path.cwd() / self.name) if not self.out_dir else self.out_dir        

    def render(self, filename: str = ""):
        dag = nx.DiGraph()
        if not self.tasks:
            print(error_msg("Skipping rendering empty workflow..."))
            return
        for task in self.tasks:
            metadata = task.metadata()
            if task.name not in dag:
                dag.add_node(task.name, item=task)
            node = dag.nodes[task.name]
            node["item"] = task
            for input_name in task.input:
                dag.add_edge(input_name, task.name)
            for output_name in task.output:
                dag.add_edge(task.name, output_name)
        dot_str = nx.nx_pydot.to_pydot(dag).to_string()
        filename = filename if filename else f"{self.name}_DAG"
        graphviz.Source(dot_str).render(
            filename=f"{self.name}_DAG", format="svg", cleanup=True
        )
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
            "file_format": self.file_format,
            "description": self.description,
        }

        for task in self.tasks:
            task_key: str = str(task.name)
            config[task_key] = {
                "type": f"{task.__class__.__module__}.{task.__class__.__name__}"
            }
            # Check if task has attribute model
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
            "file_format": self.file_format,
        }

    def _get_existing_task_ids(self) -> list[str]:
        """
        Get a list of existing task IDs in the workflow.

        This method returns a list of task IDs that are already present in the workflow.
        """
        return [task.name for task in self.tasks]

    def run(self, task_name: str, config: dict = dict()):
        task = self._get_task(task_name)        
        print(processing_msg(f"Running task: {task.name}"))            
        config = {
            **config,
            "file_format": self.file_format,
            "task": task.name,
            "input": task.input or [],
            "output": task.output or [],
            "as_dir": self.as_dir,
        }
        task.set_config(config)
        task.run()            

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
            "file_format": self.file_format,
            "task": task.name,
            "inputs": task.input or [],
            "outputs": task.output or [],
            "as_dir": self.as_dir,
        }                
        task.set_config(task_config)        
        model_path = task.train()
        task.set_model(model_path)        