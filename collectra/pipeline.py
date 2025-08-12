from rich import print
from pathlib import Path
from typing import Optional
import yaml, os, zipfile, shutil
from .task import Task, ObjectDetection, TextClassification

class TaskManager:
    """
    Manages tasks in the Collectra workflow.
    """

    VALID_TASKS = {
        "object_detection": ObjectDetection,
        "text_classification": TextClassification,
    }

    @staticmethod
    def build(task: str, from_file: bool = False, valid_tasks: list = []) -> Task:
        """
        Build a Task from a string.
        :param task: The task string in the format "<task>,<task_type>,<engine_path>,<engine_type>".        
        :raises ValueError: If the task format is invalid or the task type is not recognized
        :return: An instance of Task.
        """
        if not from_file:
            task_name, task_type, engine_path, engine_type = task.split(",")
            if task_type not in TaskManager.VALID_TASKS:
                raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TaskManager.VALID_TASKS.keys())}.")
            task = TaskManager.VALID_TASKS.get(task_type)(task_name, engine_path=engine_path, engine_type=engine_type)
            return task
        else:                                                    
            loc_task = valid_tasks.index(task["type"])
            if loc_task == -1:
                raise ValueError(f"[bold red]Invalid task type[/bold red]: {task['type']}. Must be one of {valid_tasks}.")
            TaskClass = list(TaskManager.VALID_TASKS.values())[loc_task]            
            task = TaskClass(
                id=task.get("id"),
                engine_path=task.get("engine"),
                engine_type=task.get("engine_type"),
                inputs=task.get("inputs", []),
                outputs=task.get("outputs", []),
            )
            return task            

class Collectra:
    
    def __init__(
            self,
            name: str,
            version: str,            
            file_format: Optional[str] = None,            
            description: Optional[str] = "Collectra workflow configuration",
            config: Optional[dict] = None, 
            **kwargs       
        ):
        self.name = name
        self.version = version
        self.file_format = file_format or name.lower()  # Default to workflow name if not specified        
        self.description = description
        self.tasks: list[Task] = []  # Initialize an empty list for tasks
        if config:
            self.load_config(config)
        self.as_dir = kwargs.get("as_dir", False)

    def load_config(self, config: dict):
        """
        Load the configuration from a dictionary.
        
        This method populates the workflow with tasks based on the provided configuration.
        :param config: A dictionary containing the workflow configuration.
        """
        valid_tasks = [task_class.__name__ for task_class in TaskManager.VALID_TASKS.values()]
        for task_id, task_info in config.items():
            print(task_info)
            task = {
                "id": task_id,
                **task_info
            }
            self.tasks.append(TaskManager.build(task, from_file=True, valid_tasks=valid_tasks))    
            print(f"[green]Added task:[/green] {task_id} of type {task_info['type']} with engine {task_info['engine']}")    
        print(f"[green]Loaded {len(self.tasks)} tasks from configuration.[/green]")
            

    def get_yaml(self):
        """
        Generate a YAML representation of the workflow configuration.
        
        This method serializes the workflow configuration into a YAML format.
        """     
        config = {
            "metadata": {
                "name": self.name,
                "version": self.version,            
                "file_format": self.file_format,                
                "description": self.description,            
            }
        }
    
        for task in self.tasks:
            config[task.id] = {
                "type": f"{task.__class__.__module__}.{task.__class__.__name__}",
                "engine": str(task.engine.name),
                "engine_type": task.engine.type,
                "inputs": task.inputs,
                "outputs": task.outputs,
            }

        return config

    def get_metadata(self):
        """
        Retrieve the metadata of the current workflow.
        
        This method returns the metadata of the workflow as a dictionary.
        """
        return {
            "name": self.name,
            "version": self.version,
            "file_format": self.file_format      
        }    
    
    def add(self, task: str):
        """
        Add a task to the workflow.
        
        This method parses the task string and adds it to the workflow.
        :param task: The task string in the format "<task_type>,<task_name>,<engine_type>,<engine_name>".
        """
        self.tasks.append(TaskManager.build(task))
    
    def train(self, task_id: str, config: dict = {}):
        """
        Train the specified task in the workflow.
        
        This method retrieves the task by its ID and calls its train method.
        :param task_id: The ID of the task to be trained.
        """
        config = {
            **config,
            "file_format": self.file_format,
        }
        for task in self.tasks:
            if task.id == task_id:
                print(f"[green]Training task:[/green] {task.id}")   
                task.config.update(config)  # Update task config if provided             
                task.train()
                return
        print(f"[red]Task with ID {task_id} not found.[/red]")
    
    @staticmethod
    def save(pipeline: 'Collectra', **kwargs):
        """
        Save the current pipeline configuration to the specified output directory.        
        This method serializes the pipline configuration and writes it to a file.
        """        
        as_dir = kwargs.get("as_dir", pipeline.as_dir)        
        output_dir = kwargs.get("output", Path.cwd())
        output_folder = output_dir / f"{pipeline.name}"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / "pipeline.yaml"
        config_file = pipeline.get_yaml()
        with open(output_file, 'w') as f:
            yaml.dump(config_file, f, default_flow_style=False, sort_keys=False)
        if not as_dir:
            with zipfile.ZipFile(output_dir / f"{pipeline.name}.collectra", 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_folder):
                    for file in files:
                        print(file)
                        zipf.write(os.path.join(root, file), 
                                os.path.relpath(os.path.join(root, file), 
                                                os.path.join(output_folder, '..')))        
                if not pipeline.as_dir:  
                    # If the pipeline was originally created as a directory, don't delete it          
                    shutil.rmtree(output_folder, ignore_errors=True)  # Clean up temporary files
        print(f"Workflow {pipeline.name}.collectra saved to {output_dir}")            
    
    @staticmethod
    def make(
        name: str = "default",
        version: str = "1.0",
        output: Path = Path.cwd(),
        file_format: Optional[str] = None,
        **kwargs
    ):
        """
        Create a new Collectra workflow instance.
        
        This method initializes a new workflow with default parameters.
        """
        pipeline = Collectra(name=name, version=version, file_format=file_format, **kwargs)        
        Collectra.save(pipeline, output=output)
    
    @staticmethod
    def load(pipeline: Path) -> 'Collectra':
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.
        """        
        pipeline_file = pipeline / "pipeline.yaml"                         
        tmp_path = Path("tmp")
        try:                        
            if pipeline.is_dir():            
                if not pipeline_file.exists():
                    raise FileNotFoundError(f"[red]Pipeline file not found:[/red] {pipeline_file}")
                return Collectra.load_yaml(pipeline_file, as_dir=True)
            
            if not pipeline.suffix == ".collectra":
                raise ValueError(f"[red]Invalid file format[/red]: {pipeline.suffix}. Must be a .collectra file.")
                        
            with zipfile.ZipFile(pipeline, 'r') as zipf:
                zipf.extractall(member=pipeline_file, path=tmp_path)  # Extract to a temporary directory
                return Collectra.load(tmp_path / pipeline_file)  # Load from the extracted file   
        except Exception as e:
            print(f"Error loading workflow from {pipeline}: {e}")
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)  # Clean up temporary files 
    
    @staticmethod
    def load_yaml(path: Path, **kwargs) -> 'Collectra':
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.
        """        
        as_dir = kwargs.get("as_dir", False)
        print(f"Loading workflow from {path} as_dir={as_dir}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            metadata = data.get("metadata")
            data.pop("metadata", None)  # Remove metadata from the main config dictionary            
            return Collectra(**metadata, config=data, as_dir=as_dir)  # Create a Collectra instance with the loaded data
    

