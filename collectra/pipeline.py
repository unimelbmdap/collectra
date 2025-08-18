
from rich import print
from pathlib import Path
from typing import Optional, List, Dict
import yaml, os, zipfile, shutil
from .task import Task, MachineLearningTask, ObjectDetectionYOLO, TextClassification
from .utils import error_msg, success_msg, processing_msg
from datetime import datetime
import pytz

class TaskManager:
    """
    Manages tasks in the Collectra workflow.
    """

    VALID_TASKS = [ObjectDetectionYOLO, TextClassification]    

    @staticmethod
    def get_valid_tasks() -> List[str]:
        """
        Get a list of valid task names.
        """
        return [task.__name__ for task in TaskManager.VALID_TASKS]

    @staticmethod
    def build(task: str | Dict) -> Task:
        """
        Build a Task from a string.
        :param task: The task string in the format "<task>,<task_type>,<model_path>,<model_type>".        
        :raises ValueError: If the task format is invalid or the task type is not recognized
        :return: An instance of Task.
        """
        valid_tasks = TaskManager.get_valid_tasks()
        input, output = [], []  # Initialize input and output as empty lists
        task_name, task_type, model_path = "", "", ""  # Initialize variables for task name, type, and model path
        if isinstance(task, str):        
            task_name, task_type, model_path = task.split(",")                                    
        else:
            task_name, task_type, model_path = task.get("id"), task["type"].replace("collectra.task.", ""), task.get("model")
            input, output = task.get("input", None), task.get("output", None)
            input = [input] if isinstance(input, str) else input
            output = [output] if isinstance(output, str) else output
        if task_type not in valid_tasks:
            raise ValueError(f"[bold red]Invalid task type[/bold red]: {task_type}. Must be one of {valid_tasks}."  )
        task_class_index = valid_tasks.index(task_type)    
        TaskClass = TaskManager.VALID_TASKS[task_class_index]
        return TaskClass(
            id=task_name, 
            model=model_path,
            input=input if input else [],
            output=output if output else [],
        )        

class Collectra:
    
    def __init__(
            self,
            name: str,
            version: str,            
            file_format: str = "",            
            description: str = "Collectra workflow configuration",
            config: Optional[dict] = None, 
            **kwargs       
        ):
        self.name: str = name
        self.version: str = version
        self.file_format: str = file_format or name.lower()  # Default to workflow name if not specified        
        self.description: str = description
        self.tasks: list[Task] = []  # Initialize an empty list for tasks        
        self.as_dir: bool = kwargs.get("as_dir", False)
        self.workflow_directory: Path = Path.cwd() / name
        if config:
            self.load_config(config)

    def load_config(self, config: Dict) -> None:
        """
        Load the configuration from a dictionary.
        
        This method populates the workflow with tasks based on the provided configuration.
        :param config: A dictionary containing the workflow configuration.
        """        
        for id, info in config.items():            
            task = {"id": id, **info}
            task["model"] = self.workflow_directory / task["model"] if task.get("model") else None
            task = TaskManager.build(task)            
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
            task_key: str = str(task.id)
            config[task_key] = { "type": f"{task.__class__.__module__}.{task.__class__.__name__}"}
            if isinstance(task, MachineLearningTask):                
                config[task_key]["model"] = str(task.model.path) if task.model else ""
            if task.input:                   
                config[task_key]["input"] = task.input if len(task.input) > 1 else task.input[0]  # type: ignore
            if task.output:
                config[task_key]["output"] = task.output if len(task.output) > 1 else task.output[0] # type: ignore    

        return config

    def metadata(self):
        """
        Retrieve the metadata of the current workflow.
        
        This method returns the metadata of the workflow as a dictionary.
        """
        return {
            "name": self.name,
            "version": self.version,
            "file_format": self.file_format      
        }    
    
    def add(self, task: str, task_input: str = "", task_output: List[str] = []) -> bool:
        """
        Add a task to the workflow.
        
        This method parses the task string and adds it to the workflow.
        :param task: The task string in the format "<task_type>,<task_name>,<model_type>,<model_name>".
        """
        print("[green]Adding task:[/green]", task)                
        built_task: Task = TaskManager.build(task)
        for existing_task in self.tasks:
            if existing_task.id == built_task.id:
                print(error_msg(f"Task with ID {built_task.id} already exists in the workflow."))
                print(f"Use [purple]collectra edit[/purple] command to modify the task.")                
                return False    
        self.tasks.append(built_task)
        return True

    def run(self, task_id: str, config: dict = {}):
        found_task = False
        for task in self.tasks:
            if found_task:
                break
            if task.id != task_id:
                continue
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
            if not results:
                continue
            for result in results:     
                image_file = Path(result.get("image"))  # type: ignore 
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
                if not classification_results:    
                    continue                        
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
                path = Path(f"output/{image_file.stem}.{self.file_format}")
                path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists                            
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
                    task.model = task.load(new_model_path, task.model.type)                            
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
            if not isinstance(task, MachineLearningTask):           
                continue
            model = task.get_model()                
            if not model:
                print(error_msg(f"Task {task.id} does not have a model associated with it. Skipping..."))
                continue
            model_path = model.get_path()
            if not model_path.is_file():
                print(error_msg(f"Model path does not point to a valid file: {model_path}. Skipping..."))
                continue
            if model_path != output_folder / model_path.name:
                print(processing_msg(f"Copying model {model_path} to {output_folder / model_path.name}"))
                shutil.copy(model_path, output_folder / model_path.name)
                print(processing_msg(f"Removing unneeded artifact: {model_path}"))
                os.remove(model_path)  # Remove the original file after copying
                if task.old_model and task.old_model.path.name != model_path.name:
                    print(processing_msg(f"Removing old model file: {task.old_model.path}"))
                    os.remove(task.old_model.path)  # Remove the old model if it exists
                task.model.path = Path(model_path.name)  # Update the model path to the new path

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
    def load(pipeline: Path) -> "Collectra":
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.
        """               
        if pipeline.is_dir():  # If the pipeline is a directory, load the YAML file directly
            pipeline_file = pipeline / "pipeline.yaml"  # Get the pipeline configuration file
            if not pipeline_file.exists():
                raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
            return Collectra._load_yaml(pipeline_file, as_dir=True)

        if not pipeline.suffix == ".collectra":
            raise ValueError(f"Invalid file format: {pipeline.suffix}. Must be a .collectra file.")

        pipeline_file = f"{pipeline.name.replace(pipeline.suffix, '')}/pipeline.yaml"  # Extract the pipeline configuration file name
        with zipfile.ZipFile(pipeline, 'r') as zipf:
            tmp_path = Path("tmp")  # Temporary directory to extract the pipeline
            zipf.extractall(members=[pipeline_file], path=tmp_path)  # Extract to a temporary directory                
        return Collectra._load_yaml(tmp_path / pipeline_file)  # Load from the extracted file                           

    @staticmethod
    def _load_yaml(path: Path, as_dir: bool = False) -> 'Collectra':
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.

        :param path: The path to the YAML file containing the workflow configuration.
        :param as_dir: If True, the workflow is a directory instead of a file. A flag is saved in the Collectra instance.

        """                                
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            metadata = data.get("collectra_pipeline_metadata", None)
            if not metadata:
                raise ValueError(f"Invalid pipeline file: {path}. Missing 'collectra_pipeline_metadata' section.")
            data.pop("collectra_pipeline_metadata", None)  # Remove metadata from the main config dictionary for easier loading
            return Collectra(**metadata, config=data, as_dir=as_dir)  # Create a Collectra instance with the loaded data
    

    