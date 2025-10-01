"""Data handling and persistence classes for Collectra workflows.

This module provides specialized handlers for saving workflow configurations
and artifacts to different storage formats. It supports both directory-based
and zip-based storage options with proper model artifact management.

The handlers manage:
    - Workflow configuration serialization
    - Machine learning model artifact storage
    - File system operations for different output formats
    - Model versioning and cleanup operations

Classes:
    DataHandler: Abstract base class for data handling operations
    DirectoryHandler: Handler for directory-based workflow storage
    ZipHandler: Handler for zip-based workflow storage
"""

from collectra.pipelines.base import Collectra
from collectra.tasks.base import Task
from collectra.tasks.ml import MachineLearningTask
from collectra.utils import error_msg, processing_msg, success_msg
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import os, shutil, tempfile, yaml, zipfile


@dataclass(kw_only=True)
class DataHandler:
    """Abstract base class for workflow data handling operations.
    
    Provides common functionality for saving workflow configurations and
    managing machine learning model artifacts across different storage formats.
    
    Attributes:
        pipeline (Collectra): The workflow pipeline to save.
    """

    pipeline: Collectra

    def save(self):
        """Save the workflow configuration and artifacts.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "If this is being called, the incorrect handler is not assigned."
        )

    def get_models(self, task: MachineLearningTask) -> tuple[str, str]:
        """Extract current and previous model paths from a machine learning task.
        
        Args:
            task (MachineLearningTask): The ML task to extract models from.
            
        Returns:
            tuple[str, str]: Current model path and old model path.
        """
        model, old_model = "", ""
        model = task.get_model()
        old_model = task.get_old_model()
        return model, old_model

    def is_machine_learning_task(self, task: Task | MachineLearningTask) -> bool:
        """Check if a task is a machine learning task.
        
        Args:
            task (Task | MachineLearningTask): The task to check.
            
        Returns:
            bool: True if the task is a machine learning task, False otherwise.
        """
        if not isinstance(task, MachineLearningTask):
            error_message = (
                f"Task {task.name} is not a machine learning task. Skipping..."
            )
            print(error_msg(error_message))
            return False
        return True

    def _save_config(self, tmp_dir: Path) -> None:
        """Save the workflow configuration to a YAML file.
        
        Args:
            tmp_dir (Path): Temporary directory to save the configuration file.
        """
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
class DirectoryHandler(DataHandler):
    """Handler for saving workflows as directory structures.
    
    Saves workflow configurations and artifacts to a directory structure,
    making them easy to inspect and modify manually.
    """

    def save(self):
        """Save the workflow and its artifacts to a directory structure.
        
        Creates a temporary directory for staging files, then moves them to
        the final output directory location.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_dir = self.pipeline.get_out_dir()
            tmp_dir = Path(tmpdirname)
            self._save_artifacts(tmp_dir)
            self._save_config(tmp_dir)
            out_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
            for item in tmp_dir.iterdir():
                shutil.move(item, out_dir / item.name)
            success_message = f"Workflow {self.pipeline.name} was saved successfully to {out_dir}"
            print(success_msg(success_message))

    def _save_artifacts(self, tmp_dir: Path) -> None:
        """Save machine learning model artifacts to the temporary directory.
        
        Processes all machine learning tasks in the pipeline to save their model
        artifacts. Handles model versioning, file movement, and cleanup of old models.
        
        Args:
            tmp_dir (Path): Temporary directory to save artifacts to.
        """
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
class ZipHandler(DataHandler):
    """Handler for saving workflows as compressed zip files.
    
    Saves workflow configurations and artifacts to a single zip file,
    making them portable and easy to distribute.
    """

    def save(self):
        """Save the workflow and its artifacts to a compressed zip file.
        
        Creates a temporary directory for staging files, then compresses
        them into a single zip file at the output location.
        
        Raises:
            ValueError: If the output directory is not specified.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            if not self.pipeline.out_dir:
                raise ValueError("Output directory is not specified.")
            tmp_dir = Path(tmpdirname)
            self._save_artifacts(tmp_dir)
            self._save_config(tmp_dir)
            with zipfile.ZipFile(
                f"{self.pipeline.out_dir}", "w", zipfile.ZIP_DEFLATED, allowZip64=True
            ) as zipf:
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)
                        processing_message = (
                            f"Saved {file} to {Path(self.pipeline.out_dir) / file}"
                        )
                        print(processing_msg(processing_message))
            success_message = f"Workflow {self.pipeline.name} was saved successfully at {self.pipeline.out_dir}"
            print(success_msg(success_message))

    def _save_artifacts(self, tmp_dir: Path) -> None:
        """Save machine learning model artifacts to the temporary directory for zip compression.
        
        Processes all machine learning tasks in the pipeline to save their model
        artifacts for zip archive creation. Handles model extraction from existing
        zip files and version management.
        
        Args:
            tmp_dir (Path): Temporary directory to save artifacts to.
            
        Raises:
            ValueError: If the output directory is not specified.
        """
        if not self.pipeline.out_dir:
            raise ValueError("Output directory is not specified.")
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
