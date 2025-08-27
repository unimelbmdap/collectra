from rich import print
from pathlib import Path
import yaml, os, zipfile, shutil
from .tasks import Task, MachineLearningTask
from .utils import error_msg, success_msg, processing_msg
from datetime import datetime
from dataclasses import dataclass, field
import pytz, tempfile, importlib, networkx as nx, graphviz

@dataclass(kw_only=True)
class Collectra:
    name: str
    version: str
    file_format: str
    description: str = "Collectra workflow configuration"
    tasks: list[Task] = field(default_factory=list)
    config: dict[str, str | bool] = field(default_factory=dict)

    def __post_init__(self):
        self.out_dir: str = str(self.config.get("out_dir", Path.cwd() / self.name))
        self.as_dir: bool = self.config.get("as_dir", False)  # type: ignore
        if self.as_dir:
            self.save_manager = CollectraDirectoryHandler(pipeline=self)
        else:
            self.save_manager = CollectraZipHandler(pipeline=self)

    @classmethod
    def make(
        cls,
        name: str = "default",
        version: str = "1.0",
        file_format: str = "",
        **kwargs,
    ) -> "Collectra":
        """
        Create a new Collectra workflow instance.

        This method initializes a new workflow with default parameters.
        """
        pipeline = cls(
            name=name, version=version, file_format=file_format, config=kwargs
        )
        return pipeline

    def render(self, filename: str = ""):
        dag = nx.DiGraph()
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

    def setup(self, config: dict) -> None:
        """
        Load the configuration from a dictionary.
        This method populates the workflow with tasks based on the provided configuration.
        :param config: A dictionary containing the workflow configuration.
        """
        for name, data in config.items():
            if data.get("model", None) is not None:
                data["model"] = f"{self.out_dir}/{data.get('model')}"                
                data["old_model"] = data["model"]
            if data.get("input", None) is not None:
                data["input"] = (
                    [data["input"]] if isinstance(data["input"], str) else data["input"]
                )
            if data.get("output", None) is not None:
                data["output"] = (
                    [data["output"]]
                    if isinstance(data["output"], str)
                    else data["output"]
                )
            data["name"] = name
            task_instance = TaskManager.build(data)
            self.tasks.append(task_instance)
            print(success_msg(f"Loaded {task_instance.name}"))

        if len(self.tasks) == 0:
            print("No tasks found in the workflow configuration.")

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

    def add(
        self, task: str, task_input: list[str] = [], task_output: list[str] = []
    ) -> "Collectra":
        """
        Add a task to the workflow.
        This method parses the task string and adds it to the workflow.
        :param task: The task string in the format "<task_name>,<task_type>,<model_path>".
        :param task_input: what kind of input the task accepts.
        :param task_output: what kind of output the task produces.
        """
        parts = task.split(",")
        if len(parts) != 3:
            raise ValueError(
                "Invalid task format. Expected format: <task_name>,<task_type>,<model_path>"
            )
        task_name, type, model_path = parts
        if self._find_task_by_name(task_name) and len(self._find_task_by_name(task_name)) > 0:
            error_msg = (
                f"Task already exists: {task_name}. Please provide a unique task name."
            )
            raise ValueError(error_msg)
        data = {
            "name": task_name,
            "type": type,
            "model": f"{self.out_dir}/{model_path}",
            "input": task_input,
            "output": task_output,
        }
        built_task = TaskManager.build(data)
        self.tasks.append(built_task)
        print(success_msg(f"Task with ID {built_task} added to the workflow."))
        return self

    def run(self, task_id: str, config: dict = dict()):
        found_task = False
        for task in self.tasks:
            if found_task:
                break
            if task.name != task_id:
                continue
            found_task = True
            print(processing_msg(f"Detecting task: {task.name}"))
            config = {
                **config,
                "file_format": self.file_format,
                "task": task.name,
                "input": task.input or [],
                "output": task.output or [],
                "as_dir": self.as_dir,
            }
            task.set_config(config)
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
                        str(path), "zip", path
                    )  # Create a zip archive of the results
                    shutil.rmtree(path)  # Remove the directory after zipping
                    os.rename(f"{path}.zip", path.parent / f"{path.name}")

        if not found_task:
            raise Exception(
                error_msg(f"Task with ID {task_id} not found in the workflow.")
            )

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
        self.save()

    def save(self):
        if not hasattr(self, "save_manager") or not self.save_manager:
            raise ValueError("No save manager found.")
        self.save_manager.save()


class TaskManager:

    @staticmethod
    def build(task: dict) -> Task:
        """
        Build a Task from a dictionary.
        :param task: The task dictionary containing the task details.
        :raises ValueError: If the task format is invalid or the task type is not recognized
        :return: An instance of Task.
        """
        if not task.get("name") or task["name"] == "":
            raise ValueError("Task name cannot be empty.")
        if not task.get("type"):
            raise ValueError("Task type is required.")
        module_name, class_name = task["type"].rsplit(".", 1)
        TaskClass = getattr(importlib.import_module(module_name), class_name)
        return TaskClass.build(**task)


@dataclass(kw_only=True)
class CollectraManager:

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
        data: dict = dict(), path: Path | None = None, as_dir: bool = False
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
        data.pop("collectra_pipeline_metadata", None)
        pipeline = Collectra.make(**metadata, as_dir=as_dir, out_dir=path)
        pipeline.setup(data)
        return pipeline


@dataclass(kw_only=True)
class CollectraHandler:

    pipeline: "Collectra"

    def get_models(self, task: MachineLearningTask) -> tuple[str, str]:
        model, old_model = "", ""
        model = task.get_model()
        old_model = task.get_old_model()
        return model, old_model

    def is_machine_learning_task(self, task: Task | MachineLearningTask) -> bool:
        if not isinstance(task, MachineLearningTask):
            error_message = (
                f"Task {task.name} is not a machine learning task. Skipping..."
            )
            print(error_msg(error_message))
            return False
        return True

    def _save_config(self, tmp_dir: Path) -> None:
        pipeline_yaml = tmp_dir / "pipeline.yaml"
        data = self.pipeline.get_config()
        with open(pipeline_yaml, "w") as f:
            for key in data:
                f.write(
                    yaml.dump(
                        {key: data[key]}, default_flow_style=False, sort_keys=False
                    )
                )
                f.write("\n")


@dataclass(kw_only=True)
class CollectraDirectoryHandler(CollectraHandler):

    def save(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_dir = Path(tmpdirname)
            self._save_artifacts(tmp_dir)
            self._save_config(tmp_dir)
            os.makedirs(
                self.pipeline.out_dir, exist_ok=True
            )  # Ensure the output directory exists
            for item in tmp_dir.iterdir():                
                shutil.move(item, Path(self.pipeline.out_dir) / item.name)
            success_message = f"Workflow {self.pipeline.name} was saved successfully to {self.pipeline.out_dir}"
            print(success_msg(success_message))

    def _save_artifacts(self, tmp_dir: Path) -> None:
        old_models: dict[str, Path] = dict()
        for task_item in self.pipeline.tasks:
            if not self.is_machine_learning_task(task_item):
                continue
            task: MachineLearningTask = task_item  # type: ignore
            model, old_model = self.get_models(task)
            if not model:
                error_message = f"Task {task.name} does not have a model associated with it. Skipping..."
                print(error_msg(error_message))
                continue
            # Ensure the model path is a Path object
            model_path = Path(model)            
            task_model_path = f"{task.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}-best.pt"  # Default model path for the task
            new_model = old_model != model            
            if not (model_path.exists() and model_path.is_file()):
                error_message = f"Model for {task.name} does not point to a valid file or doesn't exist: {model_path}. Attempting to download..."
                print(error_msg(error_message))
                task.load(model_path.name)
                model_path = Path(model_path.name)
                new_model = True
            if new_model:
                print(
                    f"Moving model {model_path} to temporary directory with new name {task_model_path}..."
                )
                shutil.move(model_path, tmp_dir / task_model_path)
                if old_model and Path(old_model).exists():
                    old_models[old_model] = Path(old_model)
                task.config["model"] = task_model_path
            else:
                old_models.pop(model, None)
                processing_message = f"Model {model_path} already exists. Copying to temporary directory..."
                print(processing_msg(processing_message))
                shutil.copy(model_path, tmp_dir / model_path.name)
                task.config["model"] = model_path.name

        for old_model in old_models:
            if old_models[old_model].exists():
                print(f"Removing old model {old_models[old_model]}...")
                os.remove(old_models[old_model])        


@dataclass(kw_only=True)
class CollectraZipHandler(CollectraHandler):

    pipeline: "Collectra"

    def save(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_dir = Path(tmpdirname)
            self._save_artifacts(tmp_dir)
            self._save_config(tmp_dir)
            pipeline_outdir = self.pipeline.out_dir
            with zipfile.ZipFile(
                f"{pipeline_outdir}", "w", zipfile.ZIP_DEFLATED, allowZip64=True
            ) as zipf:
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)
                        processing_message = (
                            f"Saved {file} to {Path(pipeline_outdir) / file}"
                        )
                        print(processing_msg(processing_message))
            success_message = f"Workflow {self.pipeline.name} was saved successfully at {pipeline_outdir}"
            print(success_msg(success_message))

    def _save_artifacts(self, tmp_dir: Path) -> None:
        for task_item in self.pipeline.tasks:
            if not self.is_machine_learning_task(task_item):
                continue
            task: MachineLearningTask = task_item  # type: ignore
            model, old_model = self.get_models(task)
            if not model:
                error_message = f"Task {task.name} does not have a model associated with it. Skipping..."
                print(error_msg(error_message))
                continue
            model_path = Path(model)
            task_model_path = (
                f"{task.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}-best.pt"
            )
            new_model: bool = old_model != model
            with zipfile.ZipFile(self.pipeline.out_dir, "r") as zipf:
                if not model_path.name in zipf.namelist():
                    task.load(model_path.name)
                    model_path = Path(model_path.name)
                    new_model = True
                if new_model:
                    print(
                        f"Moving model {model_path} to temporary directory with new name {task_model_path}..."
                    )
                    shutil.move(model_path, tmp_dir / task_model_path)
                    task.config["model"] = task_model_path
                else:
                    processing_message = f"Model {model_path} already exists. Copying to temporary directory..."
                    zipf.extract(str(model_path.name), tmp_dir)
                    print(processing_msg(processing_message))
                    shutil.copy(model_path, tmp_dir / model_path.name)
                    task.config["model"] = model_path.name        
