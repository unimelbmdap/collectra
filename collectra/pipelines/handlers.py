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

    pipeline: Collectra

    def save(self):
        raise NotImplementedError(
            "If this is being called, the incorrect handler is not assigned."
        )

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
class DirectoryHandler(DataHandler):

    def save(self):
        if not self.pipeline.out_dir:
            raise ValueError("Output directory is not specified.")
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
class ZipHandler(DataHandler):

    def save(self):
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
