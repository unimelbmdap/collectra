from rich import print
from pathlib import Path
from typing import Optional
import yaml, os, zipfile, shutil
from .task import Task, DetectObject, ClassifyImage

class TaskManager:
    """
    Manages tasks in the Collectra workflow.
    """

    VALID_TASKS = {
        "detect_object": DetectObject,
        "classify_image": ClassifyImage,
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
            workdir: Optional[Path] = Path("tmp"),
            description: Optional[str] = "Collectra workflow configuration",
            config: Optional[dict] = None        
        ):
        self.name = name
        self.version = version
        self.file_format = file_format or name.lower()  # Default to workflow name if not specified
        self.workdir = workdir
        self.description = description
        self.tasks: list[Task] = []  # Initialize an empty list for tasks
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
                "workdir": str(self.workdir),
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
            "file_format": self.file_format,
            "workdir": str(self.workdir),
        }    
    
    def add(self, task: str):
        """
        Add a task to the workflow.
        
        This method parses the task string and adds it to the workflow.
        :param task: The task string in the format "<task_type>,<task_name>,<engine_type>,<engine_name>".
        """
        self.tasks.append(TaskManager.build(task))
        Collectra.save(self)
    
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
    def save(pipeline: 'Collectra', output: Path = Path.cwd()):
        """
        Save the current pipeline configuration to the specified output directory.
        
        This method serializes the pipline configuration and writes it to a file.
        """        
        output_folder = Path(f"{pipeline.name}")
        output_folder.mkdir(parents=True, exist_ok=True)        
        output_file = output_folder / "pipeline.yaml"   
        config_file = pipeline.get_yaml()              
        with open(output_file, 'w') as f:
            yaml.dump(config_file, f, default_flow_style=False, sort_keys=False)        
        # with zipfile.ZipFile(f"{pipeline.file_format}.{pipeline.name}", 'w', zipfile.ZIP_DEFLATED) as zipf:            
        #     for root, dirs, files in os.walk(output_folder):
        #         for file in files:
        #             print(file)
        #             zipf.write(os.path.join(root, file), 
        #                     os.path.relpath(os.path.join(root, file), 
        #                                     os.path.join(output_folder, '..')))        
        print(f"Workflow saved to {output_file}")    
        # shutil.rmtree(output_folder, ignore_errors=True)  # Clean up temporary files
    
    @staticmethod
    def make(
        name: str = "default",
        version: str = "1.0",
        output: Path = Path.cwd(),
        file_format: Optional[str] = None,
    ):
        """
        Create a new Collectra workflow instance.
        
        This method initializes a new workflow with default parameters.
        """
        pipeline = Collectra(name=name, version=version, file_format=file_format)        
        Collectra.save(pipeline, output=output)
    
    @staticmethod
    def load_yaml(path: Path) -> 'Collectra':
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.
        """        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            metadata = data.get("metadata")
            data.pop("metadata", None)  # Remove metadata from the main data dictionary            
            return Collectra(**metadata, config=data)

    @staticmethod
    def load(pipeline: Path) -> 'Collectra':
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.
        """
        tmp_path = Path(f"tmp")
        try:                        
            pipeline_file = f"{pipeline.stem}/pipeline.yaml"
            return Collectra.load_yaml(pipeline_file)                                    
            # with zipfile.ZipFile(path, 'r') as zipf:
            #     zipf.extract(member=pipeline_file, path=tmp_path)
            #     with open(tmp_path / pipeline_file, 'r') as f:
            #         data = yaml.safe_load(f)
            #         return Collectra(**data["metadata"])
        except Exception as e:
            print(f"Error loading workflow from {pipeline}: {e}")
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)  # Clean up temporary files 
    

