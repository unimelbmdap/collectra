from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from collectra.cli import command
from collectra.types.images import Image, ImageCrop
from collectra.types.links import Link
from collectra.utils import change_dir

from collectra.utils import threading_locked
from ..machine_learning.training import run_training_command
from ..machine_learning.yolo import YOLOTask

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO
    from ultralytics.utils.metrics import ClassifyMetrics

__all__ = ["ImageClassifierYOLO"]


class ImageClassifierYOLO(YOLOTask):
    def prepare_training_inputs(
        self,
        processed_inputs: list,
        processed_parents: list,
        input_maps: dict[str, list],
        parent_input_maps: dict[str, list],
    ) -> list:
        """Use parent images labelled by their child Link node names."""
        labeled_parents = []
        for file_name, children in input_maps.items():
            parents = parent_input_maps.get(file_name, [])
            for child in children:
                if not isinstance(child, Link) or not child.parents:
                    continue
                parent_id = (
                    child.parents
                    if isinstance(child.parents, str)
                    else child.parents[0]
                )
                parent = next((item for item in parents if item.id == parent_id), None)
                if parent is not None:
                    parent.name = child.name
                    labeled_parents.append(parent)
        return labeled_parents

    @threading_locked()
    def run(self, *args: Image, **kwargs) -> Link:
        """Run image classification inference on the provided Image.

        Args:
            *args: A single Image (or ImageCrop) to classify.

        Returns:
            Link: A link from the predicted class node to the input image.
        """
        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        if not isinstance(args[0], Image):
            raise TypeError("Input must be an instance of Image.")
        image: Image = args[0]
        self._init_model()

        if isinstance(image, ImageCrop):
            with tempfile.TemporaryDirectory() as temp_dir:
                img = image.pil()
                img_path = Path(temp_dir) / Path(image.get_path()).name
                img.save(img_path)
                results: Results = (self.model(img_path))[0]
        else:
            results: Results = (self.model(image.get_path()))[0]

        if results.probs is None:
            raise ValueError("No classification probabilities returned.")

        top1_index = results.probs.top1
        predicted_class = results.names[top1_index]
        outputs = self.output if isinstance(self.output, list) else [self.output]
        if predicted_class not in outputs:
            raise ValueError(
                f"Predicted class {predicted_class!r} is not a configured output: "
                f"{outputs}"
            )
        print(f"Classified as: {predicted_class}")

        return Link(name=predicted_class, target=image)

    @command
    def train(
        self,
        inputs: list[str],
        model: str = "",
        output: Path | None = None,
        keep_log: bool = True,
        validation: str = "",
        exclude: str = "",
        epochs: int = 200,
        batch: int = 16,
        imgsz: int = 1280,
        early_stop: int = 50,
    ):
        """Train this YOLO image classifier."""
        return run_training_command(
            self,
            inputs,
            self._train,
            keep_log=keep_log,
            output=output,
            validation=validation,
            exclude=exclude,
            prepare_inputs=self.prepare_training_inputs,
            model=model,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            early_stop=early_stop,
        )

    def _train(self, *images: Image, **kwargs) -> ClassifyMetrics | None:
        from ultralytics.models import YOLO

        if kwargs.get("model"):
            self.model = kwargs["model"]
        self._init_model()
        log = kwargs.get("log", None)
        if log is None:
            raise ValueError(
                "Log directory must be specified in kwargs with key 'log'."
            )
        log = kwargs["base_folder"] / log
        log.mkdir(parents=True, exist_ok=True)
        print(f"Training files will be saved to: {log}")
        # Derive classes from training data, not from pipeline children nodes
        classes = sorted(set(img.name for img in images))
        kwargs.pop("classes", None)
        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance for training.")

        print(f"Training with model: {self.model.model_name}")

        validation = kwargs.get("validation", "")
        exclude = kwargs.get("exclude", "")
        train_dir, val_dir = self._prepare_assets(
            classes, log, validation, exclude, *images
        )
        return self._train_fold(train_dir, val_dir, classes, log, kwargs)

    def _train_fold(
        self,
        train: list[str] | Path,
        val: list[str] | Path,
        classes: list[str],
        log: Path,
        kwargs: dict,
        fold_count: int | None = None,
    ) -> ClassifyMetrics:
        from ultralytics.models import YOLO

        if not isinstance(self.model, YOLO):
            raise ValueError("Expected model to be a YOLO instance for training.")
        kwargs["config_file"] = self._prepare_yolo_config(log, classes, train, val)
        kwargs["log"] = f"{log.name}"
        if fold_count:
            kwargs["log"] += f"_fold_{fold_count}"
        params = self._prepare_params(**kwargs)
        with change_dir(kwargs["base_folder"]):
            results = self.model.train(**params)
        if results is None:
            raise Exception("[red]Training failed, no results returned.[/red]")
        self._reload()
        return results

    def _prepare_assets(
        self,
        classes: list[str],
        log: Path,
        validation_flag: str,
        exclude_flag: str,
        *images: Image,
    ) -> tuple[Path, Path]:
        """Prepare assets for YOLO classification training.

        Creates directory structure: log/train/{class}/ and log/val/{class}/

        Args:
            classes: List of class names.
            log: Path to store the assets for training.
            validation_flag: The value of the validation flag to identify validation images.
            exclude_flag: The value of the exclude flag to identify images to be excluded.
            *images: Image instances to be prepared.

        Returns:
            Tuple of (train_dir, val_dir) paths.
        """
        train_dir = log / "train"
        val_dir = log / "val"

        for cls in classes:
            (train_dir / cls).mkdir(parents=True, exist_ok=True)
            (val_dir / cls).mkdir(parents=True, exist_ok=True)

        for img in images:
            if exclude_flag and img.partition == exclude_flag:
                continue

            cls_name = img.name
            if cls_name not in classes:
                continue

            if img.partition == validation_flag:
                target_dir = val_dir / cls_name
            else:
                target_dir = train_dir / cls_name

            src = img.get_path()
            dst = target_dir / src.name
            if isinstance(img, ImageCrop) and img.source_parent:
                dst = target_dir / f"{src.stem}-{str(img.source_parent.id)}{src.suffix}"

            if not dst.exists():
                if isinstance(img, ImageCrop):
                    img.pil().save(dst)
                else:
                    shutil.copy(img.get_path(), dst)

        return train_dir, val_dir

    def _prepare_yolo_config(
        self,
        log: Path,
        classes: list[str],
        train: list[str] | Path,
        val: list[str] | Path,
    ) -> Path:
        """For classification, data param is the parent directory containing train/ and val/.

        Also prints distribution info.
        """
        self._check_distribution(log, classes)
        return log

    def _check_distribution(
        self,
        log: Path,
        metadata: dict | list[str],
    ) -> None:
        classes = (
            metadata
            if isinstance(metadata, list)
            else list(metadata.get("classes", []))
        )
        train_dir = log / "train"
        val_dir = log / "val"

        table = Table(title="Classification Distribution", show_lines=True)
        table.add_column(
            "Class Name", justify="left", style="green", header_style="bold green"
        )
        table.add_column(
            "Train Count", justify="right", style="red", header_style="bold red"
        )
        table.add_column(
            "Validation Count", justify="right", style="blue", header_style="bold blue"
        )

        for cls_name in classes:
            train_count = sum(
                1
                for ext in Image.image_types()
                for _ in (train_dir / cls_name).glob(f"*{ext}")
            )
            val_count = sum(
                1
                for ext in Image.image_types()
                for _ in (val_dir / cls_name).glob(f"*{ext}")
            )
            table.add_row(cls_name, str(train_count), str(val_count))

        console = Console()
        console.print(table)

    def _prepare_params(self, **kwargs) -> dict:
        import platform

        import torch

        params = {
            "name": kwargs["log"],
            "data": kwargs["config_file"],
            "project": kwargs["project"],
            "device": (
                "mps"
                if platform.system() == "Darwin"
                else "cuda" if torch.cuda.is_available() else "cpu"
            ),
            "epochs": kwargs.get("epochs", 1),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("early_stop", 50),
            "batch": kwargs.get("batch", 16),
        }
        return params
