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
            raise ValueError(
                f"Invalid task type: {task_type}. Must be one of {valid_tasks}."
            )
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
        **kwargs,
    ):
        """
        Create a new Collectra workflow instance.

        This method initializes a new workflow with default parameters.
        """
        pipeline = Collectra(
            name=name,
            version=version,
            file_format=file_format,
            out_dir=out_dir / name,
            **kwargs,
        )
        pipeline.save()

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
    def _load_directory(path: Path) -> "Collectra":
        """
        Load an existing Collectra workflow from a directory.
        This method reads the workflow configuration from the specified directory.
        :param path: The path to the directory containing the workflow configuration.
        :return: An instance of Collectra with the loaded configuration.
        """
        pipeline_file = path / "pipeline.yaml"
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
        with open(pipeline_file, "r") as f:
            data = yaml.safe_load(f)
            if not data:
                raise ValueError(f"Pipeline file is empty or invalid: {pipeline_file}")
            return CollectraManager._parse_data(
                data, path, as_dir=True
            )  # Load the YAML file from the directory

    @staticmethod
    def _load_zipfile(path: Path) -> "Collectra":
        """
        Load an existing Collectra workflow from a zip file.
        This method reads the workflow configuration from the specified zip file.
        :param path: The path to the zip file containing the workflow configuration.
        :return: An instance of Collectra with the loaded configuration.
        """
        pipeline_file = "pipeline.yaml"
        with zipfile.ZipFile(path, "r") as zipf:
            data = yaml.safe_load(zipf.read(pipeline_file))
            if not data:
                raise ValueError(f"Pipeline file is empty or invalid: {pipeline_file}")
            return CollectraManager._parse_data(data, path, as_dir=False)

    @staticmethod
    def _parse_data(
        data: Dict = {}, path: Optional[Path] = None, as_dir: bool = False
    ) -> "Collectra":
        """
        Parse the workflow configuration data and create a Collectra instance.
        :param data: the workflow configuration.
        :param as_dir: If True, the workflow is a directory instead of a file. A flag is saved in the Collectra instance.

        """
        metadata = data.get("collectra_pipeline_metadata", None)
        if not metadata:
            raise ValueError(
                f"Invalid pipeline file. Missing 'collectra_pipeline_metadata' section."
            )
        data.pop(
            "collectra_pipeline_metadata", None
        )  # Remove metadata from the main config dictionary for easier loading
        pipeline = Collectra(
            **metadata, as_dir=as_dir, out_dir=path
        )  # Create a Collectra instance with the loaded data
        pipeline.setup(data)
        return pipeline


class Collectra:

    def __init__(
        self,
        name: str,
        version: str,
        file_format: str,
        description: str = "Collectra workflow configuration",
        **kwargs,
    ):
        self.name: str = name
        self.version: str = version
        self.file_format: str = file_format
        self.description: str = description
        self.tasks: list[Task] = []
        self.out_dir: Path = kwargs.get("out_dir", Path.cwd() / self.name)
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
            self.tasks.append(
                TaskManager.build(task)
            )  # Build the task using the TaskManager
            print(
                success_msg(
                    f"Loaded {task['id']} of type {task['type']} and model {task['model']}"
                )
            )
        if len(self.tasks) > 0:
            print(success_msg(f"Found {len(self.tasks)} tasks from configuration"))

    def get_config(self, config: Dict = {}) -> Dict:
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
            task_key: str = str(task.id)
            config[task_key] = {
                "type": f"{task.__class__.__module__}.{task.__class__.__name__}"
            }
            if isinstance(task, MachineLearningTask):
                config[task_key]["model"] = str(task.model) if task.model else ""
            if task.input:
                config[task_key]["input"] = task.input if len(task.input) > 1 else task.input[0]  # type: ignore
            if task.output:
                config[task_key]["output"] = task.output if len(task.output) > 1 else task.output[0]  # type: ignore

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

    def _get_existing_task_ids(self) -> List[str]:
        """
        Get a list of existing task IDs in the workflow.

        This method returns a list of task IDs that are already present in the workflow.
        """
        return [task.id for task in self.tasks]

    def add(
        self, task: str, task_input: List[str] = [], task_output: List[str] = []
    ) -> "Collectra":
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
            raise ValueError(
                f"Task name is empty or already exists: {task_name}. Please provide a unique task name."
            )
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
            task.set_config(
                {
                    **config,
                    "file_format": self.file_format,
                    "task": task.id,
                    "input": task.input or [],
                    "output": task.output or [],
                }
            )
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
                    "specimen_sheet": {
                        "type": "Image",
                        "path": image_file.name,
                    },
                }
                classification_results = result.get("results", [])
                if not classification_results:
                    continue
                for cls_result in classification_results:
                    coordinates = cls_result.boxes.xywhn
                    names = [
                        cls_result.names[cls.item()]
                        for cls in cls_result.boxes.cls.int()
                    ]
                    for index in range(len(coordinates)):
                        x, y, w, h = coordinates[index]
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
                with open(path / "results.yaml", "w") as f:
                    for key in output_yaml:
                        f.write(
                            yaml.dump(
                                {key: output_yaml[key]},
                                default_flow_style=False,
                                sort_keys=False,
                            )
                        )
                        f.write("\n")
                cls_result.save_crop(save_dir=path)
                self.as_dir = config.get(
                    "as_dir", self.as_dir
                )  # Use the as_dir flag from the config if provided
                if not self.as_dir:
                    shutil.make_archive(
                        path, "zip", path
                    )  # Create a zip archive of the results
                    shutil.rmtree(path)  # Remove the directory after zipping
                    os.rename(f"{path}.zip", path.parent / f"{path.name}")

        if not found_task:
            raise Exception(
                error_msg(f"Task with ID {task_id} not found in the workflow.")
            )

    def train(self, task_id: str, config: dict = {}):
        """
        Train the specified task in the workflow.

        This method retrieves the task by its ID and calls its train method.
        :param task_id: The ID of the task to be trained.
        """
        task_arr = [task for task in self.tasks if task.id == task_id]
        if len(task_arr) == 0 or len(task_arr) > 1:
            raise ValueError(
                error_msg(f"Task with ID {task_id} has an issue in the pipeline: not found or duplicates.")
            )
        task = task_arr[0]        
        if not isinstance(task, MachineLearningTask):
            raise ValueError(
                error_msg(f"Task with ID {task_id} is not a machine learning task.")
            )        
        print(processing_msg(f"Training task: {task.id}"))
        task_config = {
            **config,
            "file_format": self.file_format,
            "task": task.id,
            "inputs": task.input or [],
            "outputs": task.output or [],
        }
        task.set_config(task_config)                    
        task.train()                
        self.save()

    def save(self):
        if self.as_dir:
            self._save_as_directory()
        else:
            self._save_as_file()
        print(
            success_msg(f"Workflow '{self.name}' saved successfully at {self.out_dir}")
        )

    def _save_as_directory(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_dir = Path(tmpdirname)
            self._save_data_assets(tmp_dir)
            self._save_pipeline_config(tmp_dir)
            os.makedirs(
                self.out_dir, exist_ok=True
            )  # Ensure the output directory exists
            for item in tmp_dir.iterdir():
                shutil.move(item, self.out_dir / item.name)

    def _save_as_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_dir = Path(tmpdirname)
            self._save_data_assets(tmp_dir)
            self._save_pipeline_config(tmp_dir)
            with zipfile.ZipFile(
                f"{self.out_dir}", "w", zipfile.ZIP_DEFLATED, allowZip64=True
            ) as zipf:
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        print(
                            processing_msg(f"Saving file to zip: {self.out_dir / file}")
                        )
                        zipf.write(os.path.join(root, file), file)
            print(
                success_msg(
                    f"Workflow '{self.name}' saved successfully at {self.out_dir}"
                )
            )

    def _save_data_assets(self, tmp_dir: Path):
        for task in self.tasks:
            new_model = False
            # First get the path of the model associated with the task
            model = task.get_model()
            if not model or not isinstance(task, MachineLearningTask):
                # If the task does not have a model, skip it
                print(
                    error_msg(
                        f"Task {task.id} does not have a model associated with it or this is not a machine learning task. Skipping..."
                    )
                )
                continue
            # Ensure the model path is a Path object
            task_model_path = f"{task.id}-best.pt"  # Default model path for the task
            model_path = self.out_dir / model
            if self.as_dir and (not model_path.exists() or not model_path.is_file()):
                # If the model path does not exist or is not a file, print an error message and attempt to load it
                print(
                    error_msg(
                        f"Model path does not point to a valid file or doesn't exist: {model_path}. Attempting to download..."
                    )
                )
                # If the model path does not exist, attempt to load it
                task.load(model)
                model_path = model
                new_model = True
            else:
                zipf = zipfile.ZipFile(self.out_dir, "r")
                if not model in zipf.namelist():
                    task.load(model)
                    model_path = model
                    new_model = True
                zipf.close()
            if new_model:
                shutil.move(
                    model_path, tmp_dir / task_model_path
                )  # Move the model to the output directory
            else:
                print(
                    processing_msg(
                        f"Model {model_path} already exists. Copying to temporary directory..."
                    )
                )
                if self.as_dir:
                    shutil.copy(model_path, tmp_dir / task_model_path)
                else:
                    zipf = zipfile.ZipFile(self.out_dir, "r")
                    zipf.extract(
                        str(model), tmp_dir
                    )  # Extract the model to the temporary directory
                    zipf.close()
            task.model = task_model_path  # Update the model path to the new path

    def _save_pipeline_config(self, tmp_dir: Path):
        pipeline_yaml = tmp_dir / "pipeline.yaml"
        data = self.get_config()
        with open(pipeline_yaml, "w") as f:
            for key in data:
                f.write(
                    yaml.dump(
                        {key: data[key]}, default_flow_style=False, sort_keys=False
                    )
                )
                f.write("\n")        
