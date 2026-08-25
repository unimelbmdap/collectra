from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image as PillowImage

from collectra.cli import command
from collectra.types.images import Image, ImageCrop
from collectra.types.links import Link
from collectra.utils import change_dir

from collectra.logger import get_logger
from collectra.utils import threading_locked
from ..machine_learning.training import print_distribution_table, run_training_command
from ..machine_learning.yolo import YOLOTask

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO
    from ultralytics.utils.metrics import ClassifyMetrics

__all__ = ["ImageClassifierYOLO"]

logger = get_logger(__name__)


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
            artefacts = {item.id: item for item in [*parents, *children]}
            for item in artefacts.values():
                if type(item) is Link:
                    item.bind(artefacts)
            for child in children:
                if type(child) is not Link:
                    raise TypeError(
                        f"Classifier label {child.id!r} in {file_name!r} must be "
                        f"a Link, got {type(child).__name__}"
                    )
                if not child.parents:
                    raise ValueError(
                        f"Classifier Link {child.id!r} in {file_name!r} has no parent"
                    )
                parent_id = (
                    child.parents
                    if isinstance(child.parents, str)
                    else child.parents[0]
                )
                parent = artefacts.get(parent_id)
                if parent is None:
                    raise ValueError(
                        f"Classifier Link {child.id!r} in {file_name!r} points to "
                        f"missing parent {parent_id!r}; available parent IDs: "
                        f"{[item.id for item in parents]}"
                    )
                child.target = parent
                try:
                    target = child.resolve()
                except RuntimeError as error:
                    raise ValueError(
                        f"Classifier Link {child.id!r} in {file_name!r} cannot "
                        f"resolve parent chain: {error}"
                    ) from error
                if not isinstance(target, Image):
                    raise TypeError(
                        f"Classifier Link {child.id!r} resolves to "
                        f"{type(target).__name__}, expected Image or ImageCrop"
                    )
                child.partition = target.partition
                labeled_parents.append(child)
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
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 1280,
        early_stop: int = 50,
        min_size: int = 0,
        weight_decay: float = 0.0005,
        dropout: float = 0.0,
        erasing: float = 0.4,
        auto_augment: str = "randaugment",
        fliplr: float = 0.5,
        flipud: float = 0.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        cls_pw: float = 0.0,
        freeze: int | None = None,
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
            min_size=min_size,
            weight_decay=weight_decay,
            dropout=dropout,
            erasing=erasing,
            auto_augment=auto_augment,
            fliplr=fliplr,
            flipud=flipud,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            cls_pw=cls_pw,
            freeze=freeze,
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
            classes,
            log,
            validation,
            exclude,
            *images,
            min_size=int(kwargs.get("min_size", 0)),
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
        min_size: int = 0,
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

        if min_size < 0:
            raise ValueError("min_size cannot be negative")

        for cls in classes:
            (train_dir / cls).mkdir(parents=True, exist_ok=True)
            (val_dir / cls).mkdir(parents=True, exist_ok=True)

        source_images: dict[Path, PillowImage.Image] = {}
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

            target = img.resolve() if type(img) is Link else img
            src = target.get_path()
            dst = target_dir / src.name
            pixel_box = None
            if isinstance(target, ImageCrop):
                dst = target_dir / f"{src.stem}-{img.id}{src.suffix}"
                left, upper, right, bottom = target.coordinates()
                pixel_box = (left, upper, right, bottom)
                width = right - left
                height = bottom - upper
            elif min_size:
                width = int(target.width)
                height = int(target.height)

            if type(img) is Link:
                dst = target_dir / f"{src.stem}-{img.id}{src.suffix}"

            empty_crop = isinstance(target, ImageCrop) and (width <= 0 or height <= 0)
            below_minimum = min_size and (width < min_size or height < min_size)
            if empty_crop or below_minimum:
                effective_minimum = min_size or 1
                parent_id = img.parent_id if type(img) is Link else ""
                logger.warning(
                    "Skipping classifier image %s%s from %s: image is %dx%d "
                    "pixels (minimum %d)%s",
                    img.id,
                    f" -> {parent_id}" if parent_id else "",
                    img.source_file or src,
                    width,
                    height,
                    effective_minimum,
                    f", pixel_box={pixel_box}" if pixel_box else "",
                )
                continue

            if not dst.exists():
                if isinstance(target, ImageCrop):
                    source_path = src.resolve()
                    if source_path not in source_images:
                        with PillowImage.open(source_path) as source:
                            source.load()
                            source_images[source_path] = source.copy()
                    with source_images[source_path].crop(pixel_box) as cropped:
                        rotated = cropped.rotate(
                            target.orientation.to_degree(), expand=True
                        )
                        try:
                            rotated.save(dst)
                        finally:
                            rotated.close()
                else:
                    shutil.copy(src, dst)

        for source in source_images.values():
            source.close()

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

        train_counts = {}
        validation_counts = {}
        for cls_name in classes:
            train_counts[cls_name] = sum(
                1
                for ext in Image.image_types()
                for _ in (train_dir / cls_name).glob(f"*{ext}")
            )
            validation_counts[cls_name] = sum(
                1
                for ext in Image.image_types()
                for _ in (val_dir / cls_name).glob(f"*{ext}")
            )
        print_distribution_table(
            "Classification Distribution", classes, train_counts, validation_counts
        )

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
            "epochs": kwargs.get("epochs", 100),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("early_stop", 50),
            "batch": kwargs.get("batch", 16),
            "weight_decay": kwargs.get("weight_decay", 0.0005),
            "dropout": kwargs.get("dropout", 0.0),
            "erasing": kwargs.get("erasing", 0.4),
            "auto_augment": kwargs.get("auto_augment", "randaugment"),
            "fliplr": kwargs.get("fliplr", 0.5),
            "flipud": kwargs.get("flipud", 0.0),
            "hsv_h": kwargs.get("hsv_h", 0.015),
            "hsv_s": kwargs.get("hsv_s", 0.7),
            "hsv_v": kwargs.get("hsv_v", 0.4),
            "cls_pw": kwargs.get("cls_pw", 0.0),
        }
        if kwargs.get("freeze") is not None:
            params["freeze"] = kwargs["freeze"]
        return params
