
from rich import print
from pathlib import Path
from typing import Optional, List, Dict
import yaml, os, zipfile, shutil    
from .task import Task, MachineLearningTask, ObjectDetectionYOLO, TextClassification
from .utils import error_msg, success_msg, processing_msg
from datetime import datetime
import pytz, tempfile

class TaskManager:

    VALID_TASKS = [ 
        ObjectDetectionYOLO, 
        TextClassification,
    ]

    """
    Manages tasks in the Collectra workflow.
    """
    @staticmethod
    def get_valid_tasks() -> List[str]:
        """
        Get a list of valid task names.
        """
        return [task.__name__ for task in TaskManager.VALID_TASKS]

    @staticmethod
    def build(task: Dict) -> Task:
        """
        Build a Task from a dictionary.
        :param task: The task dictionary containing the task details.
        :raises ValueError: If the task format is invalid or the task type is not recognized
        :return: An instance of Task.
        """        
        task_name = task.get("id", "")
        task_type = task.get("type", "").replace("collectra.task.", "")
        model_path = task.get("model")
        input = task.get("input", [])
        output = task.get("output", [])        

        if not task_name or task_name == "":
            raise ValueError("Task name cannot be empty.")        
        valid_tasks = TaskManager.get_valid_tasks()
        if task_type not in valid_tasks:
            raise ValueError(f"Invalid task type: {task_type}. Must be one of {valid_tasks}.")        
        TaskClass = TaskManager.VALID_TASKS[valid_tasks.index(task_type)]
        return TaskClass(
            id=task_name, 
            model=model_path,
            input=input,
            output=output,
        )        


class CollectraManager:

    @staticmethod
    def make(
        name: str = "default",
        version: str = "1.0",        
        file_format: str = "",
        out_dir: Path = Path.cwd(),
        **kwargs
    ) -> "Collectra":
        """
        Create a new Collectra workflow instance.
        
        This method initializes a new workflow with default parameters.
        """
        pipeline = Collectra(
            name=name, 
            version=version, 
            file_format=file_format, 
            out_dir=out_dir / name,
            **kwargs
        )        
        return pipeline.save()            

    @staticmethod
    def load(pipeline: Path) -> "Collectra":
        """
        Load an existing Collectra workflow from a YAML file.
        
        This method reads the workflow configuration from the specified path.
        """                                 
        if pipeline.is_dir():
            return CollectraManager._load_directory(pipeline)
        if pipeline.is_file():
            return CollectraManager._load_zipfile(pipeline)
        raise ValueError(f"Invalid collectra file. Please provide a valid path!")
        
    @staticmethod
    def _load_directory(path: Path) -> 'Collectra':
        """
        Load an existing Collectra workflow from a directory.    
        This method reads the workflow configuration from the specified directory.
        :param path: The path to the directory containing the workflow configuration.
        :return: An instance of Collectra with the loaded configuration.
        """
        pipeline_file = path / "pipeline.yaml"
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
        with open(pipeline_file, 'r') as f:
            data = yaml.safe_load(f)
            return CollectraManager._parse_data(data, path, as_dir=True)  # Load the YAML file from the directory

    @staticmethod
    def _load_zipfile(path: Path) -> 'Collectra':
        """
        Load an existing Collectra workflow from a zip file.        
        This method reads the workflow configuration from the specified zip file.
        :param path: The path to the zip file containing the workflow configuration.
        :return: An instance of Collectra with the loaded configuration.
        """
        pipeline_file = f"{path.name.replace(path.suffix, '')}/pipeline.yaml"        
        with zipfile.ZipFile(path, 'r') as zipf:                                            
            data = yaml.safe_load(zipf.read(pipeline_file))
            return CollectraManager._parse_data(data, path, as_dir=False)

    @staticmethod
    def _parse_data(data: Dict = {}, path: Optional[Path] = None, as_dir: bool = False) -> 'Collectra':
        """
        Parse the workflow configuration data and create a Collectra instance.
        :param data: the workflow configuration.
        :param as_dir: If True, the workflow is a directory instead of a file. A flag is saved in the Collectra instance.

        """                                                    
        metadata = data.get("collectra_pipeline_metadata", None)
        if not metadata:
            raise ValueError(f"Invalid pipeline file. Missing 'collectra_pipeline_metadata' section.")
        data.pop("collectra_pipeline_metadata", None)  # Remove metadata from the main config dictionary for easier loading
        pipeline = Collectra(**metadata, as_dir=as_dir, out_dir=path)  # Create a Collectra instance with the loaded data
        pipeline.setup(data)
        return pipeline


class Collectra:
    
    def __init__(
            self,
            name: str,
            version: str,            
            file_format: str,            
            description: str = "Collectra workflow configuration",
            **kwargs
        ):
        self.name: str = name
        self.version: str = version
        self.file_format: str = file_format
        self.description: str = description
        self.tasks: list[Task] = []       
        self.out_dir: Path = kwargs.get('out_dir', Path.cwd() / self.name)
        self.as_dir: bool = kwargs.get("as_dir", False)

    def __str__(self) -> str:
        return f"{self.name} v{self.version}"

    def setup(self, config: Dict) -> None:
        """
        Load the configuration from a dictionary.        
        This method populates the workflow with tasks based on the provided configuration.
        :param config: A dictionary containing the workflow configuration.
        """                
        for id, info in config.items():            
            task = {"id": id, **info}            
            self.tasks.append(TaskManager.build(task))  # Build the task using the TaskManager
            print(success_msg(f"Loaded {task['id']} of type {task['type']} and model {task['model']}"))        
        if len(self.tasks) > 0:
            print(success_msg(f"Found {len(self.tasks)} tasks from configuration"))

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
                config[task_key]["model"] = str(task.model) if task.model else ""
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
    
    def _get_existing_task_ids(self) -> List[str]:
        """
        Get a list of existing task IDs in the workflow.
        
        This method returns a list of task IDs that are already present in the workflow.
        """
        return [task.id for task in self.tasks]
    
    def add(self, task: str, task_input: List[str] = [], task_output: List[str] = []) -> "Collectra":
        """
        Add a task to the workflow.
        
        This method parses the task string and adds it to the workflow.
        :param task: The task string in the format "<task_name>,<task_type>,<model_path>".
        :param task_input: what kind of input the task accepts.
        :param task_output: what kind of output the task produces.
        """                 
        existing_task_ids: List[str] = self._get_existing_task_ids()
        task_name, task_type, model_path = task.split(",")
        task_obj: Dict = {
            "id": task_name,
            "type": task_type,
            "model": model_path,      
            "input": task_input,
            "output": task_output,      
        }
        if task_name in existing_task_ids:
            raise ValueError(f"Task name is empty or already exists: {task_name}. Please provide a unique task name.")
        built_task: Task = TaskManager.build(task_obj)                
        self.tasks.append(built_task)                                
        print(success_msg(f"Task with ID {built_task.id} added to the workflow."))        
        return self

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
        self.save()  # Save the workflow after training
    
    def save(self):
        if self.as_dir:
            self._save_as_directory()
        else:
            self._save_as_file()              
        # tmpsave = tempfile.TemporaryDirectory()
        # out_dir = Path(tmpsave.name)
        # for task in self.tasks:
        #     model_path = task.get_model()                
        #     if not model_path:
        #         print(error_msg(f"Task {task.id} does not have a model associated with it. Skipping..."))
        #         continue                                              
        #     model_exists = False
        #     if not self.as_dir:                    
        #         zipf =  zipfile.ZipFile(f"{self.out_dir}.collectra", 'r')
        #         model_exists = f"{out_dir.name}/{model_path}" in zipf.namelist()
        #         zipf.close()                    
        #     else:                    
        #         model_exists = (out_dir / model_path).is_file()
        #     new_model_path: Optional[Path] = None
        #     if isinstance(task, MachineLearningTask) and not model_exists:
        #         print(error_msg(f"Model path does not point to a valid file or doesn't exist. Attempting to download..."))
        #         task.load(model_path)
        #         new_model_path = Path(out_dir.name) / model_path                            

        # self.out_dir.mkdir(parents=True, exist_ok=True)  
        # self.__save_data_assets()
        # self.__save_pipeline_config()        
        # if not self.as_dir:
        #     self.__save_to_zip()                 
        # self.cleanup()
        # tmpsave.cleanup()        
    
    def _save_as_directory(self):
        pass

    def _save_as_file(self):
        print(self.out_dir)
        pass

    def __save_data_assets(self):
        for task in self.tasks:             
            model = task.get_model()                
            if not model:
                print(error_msg(f"Task {task.id} does not have a model associated with it. Skipping..."))
                continue
            actual_model_path = self.out_dir / model            
            if isinstance(task, MachineLearningTask) and not actual_model_path.is_file():
                print(error_msg(f"Model path does not point to a valid file or doesn't exist: {actual_model_path}. Attempting to download..."))
                task.load(model)
            if actual_model_path != self.out_dir / model:
                print(processing_msg(f"Copying model {model_path} to {output_folder / model_path.name}"))
                shutil.move(model_path, output_folder / model_path.name)                                
                if task.old_model and Path(task.old_model).name != model_path.name:
                    print(processing_msg(f"Removing old model file: {task.old_model}"))
                    os.remove(task.old_model)  # Remove the old model if it exists
                task.model = model_path.name  # Update the model path to the new path

    def __save_pipeline_config(self):
        output_file = self.out_dir / "pipeline.yaml"
        config_file = self.get_yaml()
        with open(output_file, 'w') as f:
            for key in config_file:
                f.write(yaml.dump({key: config_file[key]}, default_flow_style=False, sort_keys=False))
                f.write("\n")            

    def __save_to_zip(self):
        with zipfile.ZipFile(f"{self.out_dir}.collectra", 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            for root, dirs, files in os.walk(self.out_dir):
                for file in files:
                    print(processing_msg(f"Saving file to zip: {os.path.join(root, file)}"))
                    zipf.write(os.path.join(root, file),
                            os.path.relpath(os.path.join(root, file),
                                            os.path.join(self.out_dir, '..')))
            if not self.as_dir:                           
                shutil.rmtree(self.out_dir, ignore_errors=True)

    def cleanup(self):
        """
        Clean up temporary files created during the workflow execution.
        
        This method removes the temporary directory used for storing intermediate files.
        """
        tmp_path = Path("tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)  # Clean up temporary files   
        # Remove any .pt files in the current directory
        for file in os.listdir("."):
            if file.endswith(".pt"):
                print(processing_msg(f"Removing unneeded artifact: {file}"))
                os.remove(file)
        if not self.as_dir and self.out_dir.exists():
            # If the pipeline was not originally created as a directory, delete it            
            shutil.rmtree(self.out_dir, ignore_errors=True)                             
    

    