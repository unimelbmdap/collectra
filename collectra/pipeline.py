from rich import print
from pathlib import Path
from typing import Optional, List
import yaml, os, zipfile, shutil
from .task import Task, ObjectDetectionYOLO, TextClassification
from .utils import error_msg, success_msg, processing_msg
from datetime import datetime
import pytz

class TaskManager:
    """
    Manages tasks in the Collectra workflow.
    """

    VALID_TASKS = {
        "object_detection": ObjectDetectionYOLO,
        "text_classification": TextClassification,
    }

    @staticmethod
    def build(task: str, from_file: bool = False, valid_tasks: list = []) -> Task:
        """
        Build a Task from a string.
        :param task: The task string in the format "<task>,<task_type>,<model_path>,<model_type>".        
        :raises ValueError: If the task format is invalid or the task type is not recognized
        :return: An instance of Task.
        """
        if not from_file:
            task_name, task_type, model_path, model_type = task.split(",")
            if task_type not in TaskManager.VALID_TASKS:
                raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {list(TaskManager.VALID_TASKS.keys())}.")
            task = TaskManager.VALID_TASKS.get(task_type)(task_name, model_path=model_path, model_type=model_type)            
            return task
        else:
            task_type = task["type"].replace("collectra.task.", "")                                                  
            loc_task = valid_tasks.index(task_type)
            if loc_task == -1:
                raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {valid_tasks}.")
            TaskClass = list(TaskManager.VALID_TASKS.values())[loc_task]            

            input = task.get("input", None)
            output = task.get("output", None)

            if type(input) is str:
                input = [input]
            if type(output) is str:
                output = [output]

            task = TaskClass(
                id=task.get("id"),
                model_path=task.get("model"),
                model_type=task.get("model_type"),
                input=input if input else [],
                output=output if output else [],
            )
            print(success_msg(f"Loaded task {task.id} of type {task.__class__.__name__} and model {task.model}"))
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
            task["model"] = Path(self.working_dir) / task["model"] if task.get("model") else None
            task = TaskManager.build(task, from_file=True, valid_tasks=valid_tasks)
            if not task:
                print(error_msg(f"Failed to load task {task_id} of type {task_info['type']} and model {task_info['model']}"))
                continue
            self.tasks.append(task)
            print(success_msg(f"Loaded {task.id} of type {task.__class__.__name__} and model {task.model}"))
        print(success_msg(f"Loaded {len(self.tasks)} tasks from configuration!"))

    def get_yaml(self):
        """
        Generate a YAML representation of the workflow configuration.
        
        This method serializes the workflow configuration into a YAML format.
        """     
        config = {
            "collectra_pipeline_metadata": {
                "name": self.name,
                "version": self.version,            
                "file_format": self.file_format,                
                "description": self.description,            
            }
        }
    
        for task in self.tasks:
            config[task.id] = {
                "type": f"{task.__class__.__module__}.{task.__class__.__name__}",
                "model": str(task.model.name),
                "model_type": task.model.type,
                "input": task.input if len(task.input) > 1 else task.input[0],
                "output": task.output if len(task.output) > 1 else task.output[0],
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
    
    def add(self, task: str, task_input: str = None, task_output: List[str] = []) -> bool:
        """
        Add a task to the workflow.
        
        This method parses the task string and adds it to the workflow.
        :param task: The task string in the format "<task_type>,<task_name>,<model_type>,<model_name>".
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

    def run(self, task_id: str, config: dict = {}):
        found_task = False
        for task in self.tasks:
            if task.id == task_id:
                found_task = True
                print(processing_msg(f"Detecting task: {task.id}"))
                task.set_config({
                    **config,
                    "file_format": self.file_format,
                    "task": task.id,
                    "input": task.input or [],
                    "output": task.output or [],                    
                })                
                results = task.run()
                names = []
                if results:
                    for result in results:     
                        image_file = Path(result.get("image"))      
                        output_yaml = {
                            "collectra_results_metadata": {
                                "timestamp": datetime.now(pytz.utc).isoformat(),
                                "validation": False,
                            },
                            "specimen_sheet":{
                                "type": "Image",
                                "path": image_file.name,                                
                            },                            
                        }                 
                        classification_results = result.get("results", [])                        
                        if classification_results:                            
                            for cls_result in classification_results:                                                            
                                coordinates = cls_result.boxes.xywhn
                                names = [cls_result.names[cls.item()] for cls in cls_result.boxes.cls.int()]
                                for index in range(len(coordinates)):
                                    x,y,w,h = coordinates[index]
                                    output_yaml[names[index]] = { 
                                        "type": "ImageCrop",
                                        "image": "specimen_sheet",                                       
                                        "x_center": float(x),
                                        "y_center": float(y),
                                        "width_relative": float(w),
                                        "height_relative": float(h),
                                    }                                                            
                            path = Path(f"{image_file.stem}.{self.file_format}")
                            os.makedirs(path, exist_ok=True)
                            # Copy the image to the output directory                                                        
                            shutil.copy(image_file, path / image_file.name)  
                            with open(path / "results.yaml", 'w') as f:
                                for key in output_yaml:
                                    f.write(yaml.dump({key: output_yaml[key]}, default_flow_style=False, sort_keys=False))
                                    f.write("\n")
                            cls_result.save_crop(save_dir=path)   
                            self.as_dir = config.get("as_dir", self.as_dir)  # Use the as_dir flag from the config if provided
                            if not self.as_dir:
                                shutil.make_archive(path, 'zip', path)  # Create a zip archive of the results     
                                shutil.rmtree(path)  # Remove the directory after zipping
                                os.rename(f"{path}.zip", path.parent / f"{path.name}")

        if not found_task:
            raise Exception(error_msg(f"Task with ID {task_id} not found in the workflow."))


    def train(self, task_id: str, config: dict = {}):
        """
        Train the specified task in the workflow.
        
        This method retrieves the task by its ID and calls its train method.
        :param task_id: The ID of the task to be trained.
        """        
        found_task = False
        for task in self.tasks:
            if task.id == task_id:
                found_task = True
                print(processing_msg(f"Training task: {task.id}"))
                task.set_config({
                    **config,
                    "file_format": self.file_format,
                    "task": task.id,
                    "inputs": task.input or [],
                    "outputs": task.output or [],
                })                                                 
                new_model_path: Path = task.train()                
                if new_model_path:
                    task.old_model = task.model
                    task.model = task.validate_model(new_model_path, task.model.type)                            
                break
        if not found_task:
            raise Exception(error_msg(f"Task with ID {task_id} not found in the workflow."))            
        self.save(as_dir=self.as_dir)  # Save the workflow after training
    
    def save(self, **kwargs):
        self.as_dir = kwargs.get("as_dir", self.as_dir)
        output_dir = kwargs.get("output", Path.cwd())
        output_folder = output_dir / f"{self.name}"
        output_folder.mkdir(parents=True, exist_ok=True)
        self.__save_data_assets(output_folder=output_folder)
        self.__save_pipeline_config(output_folder=output_folder)        
        if not self.as_dir:
            self.__save_to_zip(output_folder=output_folder, output_dir=output_dir)
        print(success_msg(f"Workflow {self.name}.collectra saved to {output_dir}"))

    def __save_data_assets(self, output_folder: Path):
        for task in self.tasks:            
            model_path = task.get_model().get_path()
            if not model_path.is_file():
                print(error_msg(f"Model path does not point to a file: {model_path}"))
                continue
            if model_path != output_folder / model_path.name:
                shutil.copy(model_path, output_folder / model_path.name)                
                os.remove(model_path)  # Remove the original file after copying
                if task.old_model:
                    os.remove(output_folder / task.old_model.name)  # Remove the old model if it exists
                task.model.name = model_path.name  # Update the model name to the new path
                        
    def __save_pipeline_config(self, output_folder: Path):
        output_file = output_folder / "pipeline.yaml"
        config_file = self.get_yaml()
        with open(output_file, 'w') as f:
            for key in config_file:
                f.write(yaml.dump({key: config_file[key]}, default_flow_style=False, sort_keys=False))
                f.write("\n")            

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
            metadata = data.get("collectra_pipeline_metadata")
            data.pop("collectra_pipeline_metadata", None)  # Remove metadata from the main config dictionary
            return Collectra(**metadata, config=data, as_dir=as_dir)  # Create a Collectra instance with the loaded data
    

    