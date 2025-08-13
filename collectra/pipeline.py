from rich import print
from pathlib import Path
from typing import Optional
import yaml, os, zipfile, shutil
from .task import Task, ObjectDetection, TextClassification
from .utils import error_msg, success_msg, processing_msg

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
            task_type = task["type"].replace("collectra.task.", "")                                                  
            loc_task = valid_tasks.index(task_type)
            if loc_task == -1:
                raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {valid_tasks}.")
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
        self.as_dir = kwargs.get("as_dir", False)
        self.working_dir = Path.cwd() / kwargs.get("working_dir", name)
        if config:
            self.load_config(config)

    def load_config(self, config: dict):
        """
        Load the configuration from a dictionary.
        
        This method populates the workflow with tasks based on the provided configuration.
        :param config: A dictionary containing the workflow configuration.
        """
        valid_tasks = [task_class.__name__ for task_class in TaskManager.VALID_TASKS.values()]        
        for task_id, task_info in config.items():            
            task = {
                "id": task_id,
                **task_info
            }            
            task["engine"] = Path(self.working_dir) / task["engine"] if task.get("engine") else None
            task = TaskManager.build(task, from_file=True, valid_tasks=valid_tasks)
            if not task:
                print(error_msg(f"Failed to load task {task_id} of type {task_info['type']} and model {task_info['engine']}"))
                continue
            self.tasks.append(task)
            print(success_msg(f"Loaded {task.id} of type {task.__class__.__name__} and model {task.engine}"))
        print(success_msg(f"Loaded {len(self.tasks)} tasks from configuration!"))

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
    
    def add(self, task: str) -> bool:
        """
        Add a task to the workflow.
        
        This method parses the task string and adds it to the workflow.
        :param task: The task string in the format "<task_type>,<task_name>,<engine_type>,<engine_name>".
        """
        print("[green]Adding task:[/green]", task)                
        task = TaskManager.build(task)
        for existing_task in self.tasks:
            if existing_task.id == task.id:
                print(error_msg(f"Task with ID {task.id} already exists in the workflow."))
                print(f"Use [purple]collectra edit[/purple] command to modify the task.")
                task.delete() # Remove the task if it already exists
                return False    
        self.tasks.append(task)
        return True

    
    def train(self, task_id: str, config: dict = {}):
        """
        Train the specified task in the workflow.
        
        This method retrieves the task by its ID and calls its train method.
        :param task_id: The ID of the task to be trained.
        """        
        for task in self.tasks:
            if task.id == task_id:
                print(processing_msg(f"Training task: {task.id}"))
                task.set_config({
                    **config,
                    "file_format": self.file_format,
                    "task": task.id,
                    "inputs": task.inputs or [],
                    "outputs": task.outputs or [],
                })                
                task.train()
                return        
        print(f"{error_msg('Task not found')}: {task_id}")
    
    def save(self, **kwargs):
        as_dir = kwargs.get("as_dir", self.as_dir)
        output_dir = kwargs.get("output", Path.cwd())
        output_folder = output_dir / f"{self.name}"
        output_folder.mkdir(parents=True, exist_ok=True)
        self.__save_pipeline_config(output_folder=output_folder)
        self.__save_data_assests(output_folder=output_folder)
        if not as_dir:
            self.__save_to_zip(output_folder=output_folder, output_dir=output_dir)
        print(success_msg(f"Workflow {self.name}.collectra saved to {output_dir}"))

    def __save_data_assests(self, output_folder: Path):
        for task in self.tasks:
            engine_path = task.get_engine().get_path()
            if not engine_path.is_file():
                print(error_msg(f"Engine path does not point to a file: {engine_path}"))
                continue
            if engine_path != output_folder / engine_path.name:
                shutil.copy(engine_path, output_folder / engine_path.name)
                os.remove(engine_path)  # Remove the original file after copying
                        
    def __save_pipeline_config(self, output_folder: Path):
        output_file = output_folder / "pipeline.yaml"
        config_file = self.get_yaml()
        with open(output_file, 'w') as f:
            yaml.dump(config_file, f, default_flow_style=False, sort_keys=False)

    def __save_to_zip(self, output_folder: Path, output_dir: Path = Path.cwd()):
        with zipfile.ZipFile(output_dir / f"{self.name}.collectra", 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:                       
                    print(processing_msg(f"Saving file to zip: {os.path.join(root, file)}"))                 
                    zipf.write(os.path.join(root, file), 
                            os.path.relpath(os.path.join(root, file), 
                                            os.path.join(output_folder, '..')))      
            if not self.as_dir:  
                # If the pipeline was originally created as a directory, don't delete it          
                shutil.rmtree(output_folder, ignore_errors=True)  # Clean up temporary files        
    
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
                raise ValueError(error_msg(f"Invalid file format: {pipeline.suffix}. Must be a .collectra file."))
                        
            with zipfile.ZipFile(pipeline, 'r') as zipf:
                zipf.extractall(member=pipeline_file, path=tmp_path)  # Extract to a temporary directory
                return Collectra.load_yaml(tmp_path / pipeline_file)  # Load from the extracted file   
        except Exception as e:
            print(error_msg(f"Error loading workflow from {pipeline}: {e}"))
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)  # Clean up temporary files 
    
    @staticmethod
    def load_yaml(path: Path, **kwargs) -> 'Collectra':
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.
        """        
        as_dir = kwargs.get("as_dir", False)
        print(processing_msg(f"Loading workflow from {path} as_dir={as_dir}"))
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            metadata = data.get("metadata")
            data.pop("metadata", None)  # Remove metadata from the main config dictionary            
            return Collectra(**metadata, config=data, as_dir=as_dir)  # Create a Collectra instance with the loaded data
    

