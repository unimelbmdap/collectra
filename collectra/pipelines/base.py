"""Core pipeline execution and workflow management for Collectra.

This module contains the main Collectra workflow class and result management
system. It provides the core functionality for defining, executing, and
managing complex data processing workflows with task dependencies.

The module supports:
    - Workflow definition and configuration management
    - Task dependency graph construction and execution
    - Result aggregation and persistence
    - Workflow visualization and rendering
    - Training and inference execution modes

Classes:
    Collectra: Main workflow management class
    CollectraResult: Container for workflow execution results
"""

from collectra.tasks.base import Task, TaskResult
from collectra.tasks.ml import MachineLearningTask
from collectra.utils import error_msg, success_msg, processing_msg
from dataclasses import dataclass, field
from pathlib import Path
from rich import print
import networkx as nx, graphviz
import copy, logging, shutil, os, yaml, datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass(kw_only=True)
class Collectra:
    """Main class for managing Collectra workflows and task execution.
    
    This class represents a complete workflow configuration with tasks, dependencies,
    and execution management. It provides functionality to add tasks, connect them
    in a dependency graph, execute the workflow, and manage results.
    
    Attributes:
        name (str): The name of the workflow.
        version (str): The version of the workflow.
        format (str): The output format for results.
        out_dir (str | None): Output directory path. Defaults to None.
        as_dir (bool): Whether to treat output as a directory structure. Defaults to False.
        description (str): Description of the workflow configuration.
        tasks (list[Task]): List of tasks in the workflow.
        flow (nx.DiGraph | None): Directed graph representing task dependencies.
        results (dict): Dictionary storing workflow execution results.
        cli_kwargs (list[str]): List of CLI keyword arguments.
    """
    name: str
    version: str
    format: str
    out_dir: str | None = None
    as_dir: bool = False
    description: str = "Collectra workflow configuration"
    tasks: list[Task] = field(default_factory=list)      
    flow: nx.DiGraph | None = None
    results: dict = field(default_factory=dict)
    cli_kwargs: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the workflow after dataclass initialization.
        
        Sets up the directed graph for task dependencies and creates a result container.
        """               
        self.flow = nx.DiGraph()
        self.result = CollectraResult()

    def get_out_dir(self) -> Path:
        """Get the output directory path for the workflow.
        
        If no output directory is specified, defaults to current working directory
        with the workflow name as subdirectory.
        
        Returns:
            Path: The output directory path.
        """
        if not self.out_dir:             
            self.out_dir = str(Path.cwd() / self.name)
        return Path(self.out_dir)
    
    def get_run_result(self) -> dict:
        """Get the execution results of the workflow.
        
        Returns:
            dict: Dictionary containing the workflow execution results.
        """
        return self.results

    def is_pipeline_dir(self) -> bool:
        """Check if the pipeline output should be treated as a directory structure.
        
        Returns:
            bool: True if output should be a directory structure, False otherwise.
        """
        return self.as_dir

    def add_task(self, task: Task, mapping: dict):
        """Add a task to the workflow and update the output mapping.
        
        Args:
            task (Task): The task to add to the workflow.
            mapping (dict): Dictionary mapping outputs to task names.
            
        Raises:
            ValueError: If the pipeline is not initialized or task already exists.
        """
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
        """Connect tasks in the workflow based on input-output mapping.
        
        Args:
            mapping (dict): Dictionary mapping outputs to task names for establishing dependencies.
            
        Raises:
            ValueError: If the pipeline is not initialized.
        """   
        if self.flow is None:
            raise ValueError(f"Pipeline has not been initialised for {self.name}")  
        for task in self.tasks:
            for input in task.input:                
                parent_tasks = mapping.get(input, [])
                for parent_task in parent_tasks:
                    self.flow.add_edge(parent_task, task.name)     

    def render(self, filename: str = "", raw=False):
        """Render the workflow as a directed acyclic graph (DAG) visualization.
        
        Creates an SVG visualization of the workflow showing tasks and their dependencies.
        Can render either the raw task graph or a formatted version with inputs/outputs.
        
        Args:
            filename (str, optional): Name for the output file. Defaults to workflow name with DAG suffix.
            raw (bool, optional): Whether to render raw task graph or formatted version. Defaults to False.
            
        Raises:
            ValueError: If the pipeline is not initialized.
        """
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
        if self.is_pipeline_dir():
            graphviz.Source(dot_str).render(
                filename=f"{self.get_out_dir() / filename}", format="svg", cleanup=True
            )
        else:
            graphviz.Source(dot_str).render(filename=filename, format="svg", cleanup=True)        
        print(success_msg(f"Workflow rendered to {filename}.svg"))

    def __str__(self) -> str:
        """Return a string representation of the workflow.
        
        Returns:
            str: String containing workflow name and version.
        """
        return f"{self.name} v{self.version}"

    def get_config(self, config: dict = dict()) -> dict:
        """Generate a YAML representation of the workflow configuration.

        This method serializes the workflow configuration into a YAML format,
        including metadata and task configurations.

        Args:
            config (dict, optional): Existing configuration dictionary to update. Defaults to empty dict.

        Returns:
            dict: Dictionary containing the complete workflow configuration in YAML format.
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
            if isinstance(task.input, dict):
                task.input = list(task.input.keys())  
            if isinstance(task.output, dict):
                task.output = list(task.output.keys())  
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
        """Retrieve the metadata of the current workflow.

        This method returns the metadata of the workflow as a dictionary.
        
        Returns:
            dict: Dictionary containing workflow metadata including name, version, and format.
        """
        return {
            "name": self.name,
            "version": self.version,
            "format": self.format,
        }

    def _get_existing_task_ids(self) -> list[str]:
        """Get a list of existing task IDs in the workflow.

        This method returns a list of task IDs that are already present in the workflow.
        
        Returns:
            list[str]: List of existing task IDs in the workflow.
        """
        return [task.name for task in self.tasks]

    def __call__(self, task_name: str, output: Path, **kwargs):
        """Make the Collectra instance callable.
        
        Args:
            task_name (str): Name of the task to run.
            output (Path): Output path for results.
            **kwargs: Additional keyword arguments passed to run method.
        """ 
        self.cli_kwargs = list(kwargs.keys())      
        if output is None and 'file' in kwargs:
            output = Path(kwargs["file"])
        self.run(task_name, output=output, **kwargs)

    def run(self, task_name: str, output: Path, **kwargs):
        """Execute the workflow starting from the specified task or root tasks.
        
        Args:
            task_name (str): Name of the specific task to run. If empty, runs from root tasks.
            output (Path): Output path for saving results.
            **kwargs: Additional keyword arguments passed to task execution.
            
        Raises:
            ValueError: If the pipeline is not initialized or task not found.
        """
        if self.flow is None:
            raise ValueError(f"Pipeline is not initialised for: {self.name}")                  
        if task_name:
            initial_nodes = [node for node in self.flow.nodes() if str(node) == task_name]            
        else:
            initial_nodes = [node for node in self.flow.nodes() if len(list(self.flow.predecessors(node))) == 0]
        if task_name and not initial_nodes:
            raise ValueError(f"Task {task_name} not found in the workflow.")
        single = kwargs.pop("single", False)
        self._run_nodes(initial_nodes, single=single, **kwargs)
        if self.result.failed_tasks:
            self.result.status = "failed" if not self.result.successful_tasks else "partial"   
        if self.result.status == "partial" or self.result.status == "failed":
            for failed_task in self.result.failed_tasks:
                print(error_msg(f"Task {failed_task.task.name} failed with error:\n{failed_task.get_error()}"))     
        self.result.save(output, keys_to_remove=self.cli_kwargs, format=self.format, as_dir=self.is_pipeline_dir(), **kwargs)    

    def _run_nodes(self, nodes: list[str], single: bool=False, **kwargs):
        """Recursively execute workflow nodes and their dependencies.
        
        Args:
            nodes (list[str]): List of node names to execute.
            single (bool, optional): Whether to run only single node without children. Defaults to False.
            **kwargs: Additional keyword arguments passed to task execution.
            
        Raises:
            ValueError: If the pipeline is not initialized.
        """        
        if self.flow is None:
            raise ValueError(f"Pipeline is not initialised for: {self.name}")                 
        for node in nodes:
            try:                
                task_result: TaskResult = self._run_task(node, **kwargs)                                                                                                              
                output_data = copy.deepcopy(task_result.output)                
                self.result.successful_tasks.append(task_result)                                     
                children_nodes = list(self.flow.successors(node)) if single is False else []                        
                if children_nodes:
                    self._run_nodes(children_nodes, **output_data)                                
            except Exception as e:
                task_result = TaskResult.create_failed(self._get_task(node), e)                
                self.result.failed_tasks.append(task_result)

    def _run_task(self, task_name: str, **kwargs) -> TaskResult:
        """Execute a single task in the workflow.
        
        Args:
            task_name (str): Name of the task to execute.
            **kwargs: Additional keyword arguments passed to task execution.
            
        Returns:
            TaskResult: Result of the task execution.
        """                
        task = self._get_task(task_name)                                           
        task_config = {            
            "format": self.format,                        
            "as_dir": self.is_pipeline_dir(),
        }                        
        task.set_config(task_config)            
        print(processing_msg(f"Running task: {task.name}"))               
        return task(**kwargs)                               

    def _find_task_by_name(self, task_name: str) -> list[Task]:
        """Find tasks in the workflow by name.
        
        Args:
            task_name (str): Name of the task to find.
            
        Returns:
            list[Task]: List of tasks matching the given name.
        """
        return [task for task in self.tasks if task.name == task_name]

    def _get_task(self, task_name: str):
        """Get a single task by name from the workflow.
        
        Args:
            task_name (str): Name of the task to retrieve.
            
        Returns:
            Task: The task matching the given name.
            
        Raises:
            ValueError: If no task or multiple tasks with the same name are found.
        """
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
            config (dict, optional): Additional configuration for training. Defaults to empty dict.

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
            "as_dir": self.is_pipeline_dir(),
        }                
        task.set_config(task_config)        
        model_path = task.train()
        task.set_model(model_path)        

@dataclass(kw_only=True)
class CollectraResult:
    """Container for storing and managing workflow execution results.
    
    This class tracks successful and failed task executions and provides
    functionality to save results to disk in various formats.
    
    Attributes:
        successful_tasks (list): List of successfully executed tasks.
        failed_tasks (list): List of tasks that failed during execution.
        status (str): Overall execution status - 'success', 'partial', or 'failed'.
    """
    successful_tasks: list = field(default_factory=list)
    failed_tasks: list = field(default_factory=list)
    status: str = "success"  # success, partial, failed

    def save(self, output: Path, **kwargs) -> None:
        """Save workflow execution results to the specified output path.
        
        Aggregates results from all successful task executions and saves them
        to a YAML file along with associated output files. Creates the output
        directory if it doesn't exist.
        
        Args:
            output (Path): Directory path where results will be saved.
            **kwargs: Additional keyword arguments passed to task save methods.
            
        Raises:
            ValueError: If no successful tasks exist to save or no result path
                can be determined.
        """
        if not self.successful_tasks:
            raise ValueError("No successful tasks to save.")      
        final_result_dict = {
             "collectra_results_metadata":{
                "timestamp": datetime.datetime.now().isoformat(), 
                "validation": False
            }
        } 
        final_result_path = output
        final_result_files = set()
        for task_result in self.successful_tasks:                                                                             
            result_path, result_dict, files = task_result.task.save(output_path=output, output_data=final_result_dict, **kwargs)                            
            final_result_dict.update(result_dict)    
            final_result_files.update(files)            
            final_result_path = result_path if result_path else final_result_path        
        if not final_result_path:
            raise ValueError("No result path to save.")    
        os.makedirs(final_result_path, exist_ok=True)      
        breakpoint()
             
        with open(final_result_path / "results.yaml", "w", encoding="utf-8") as f:
            for key, value in final_result_dict.items():
                yaml_representation_str = yaml.dump({key: value}, default_flow_style=False, sort_keys=False, allow_unicode=True)
                f.write(yaml_representation_str)
                f.write("\n")                
        for file in final_result_files:
            if (final_result_path / file.name).exists():
                continue        
            shutil.copy(file, final_result_path / file.name) 
        if kwargs.get("print_log", False):
            log_message = f"""
                Results saved to {final_result_path}
                Successful tasks: {[str(task) for task in self.successful_tasks]}
                Failed tasks: {[str(task) for task in self.failed_tasks]}
            """
            print(log_message)
            if self.status == "partial":
                print(error_msg("Some tasks failed during execution."))
            elif self.status == "failed":
                print(error_msg("All tasks failed during execution."))
            else:
                print(success_msg(f"All tasks completed successfully."))
