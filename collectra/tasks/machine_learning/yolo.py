import shutil
import tempfile
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from ultralytics.engine.results import Results
from ultralytics.models import YOLO
from ultralytics.utils import ThreadingLocked
from ultralytics.utils.metrics import DetMetrics

from collectra.types.images import Image, ImageCrop
from collectra.utils import change_dir

from .base import MachineLearningTask

__all__ = ["ObjectDetectionYOLO"]


class ObjectDetectionYOLO(MachineLearningTask):

    model: str | Path | YOLO
    original_model_path: str | Path = ""

    def _init_model(self) -> None:
        """Ensure that the YOLO model is loaded before performing any operations."""
        if self.model and isinstance(self.model, (str, Path)):
            self._load(Path(self.model))
        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance")

    def _load(self, model: str | Path) -> None:
        """Load the YOLO model for object detection.

        Args:
            model (str | Path): The path to the model file or the model itself.
        """
        self.original_model_path = model
        self.model = YOLO(model)

    @ThreadingLocked()
    def run(self, *args: Image) -> list[Image]:
        """Run object detection inference on the provided Image.

        This method performs object detection on the provided input image
        using the trained YOLO model.

        Results include bounding boxes, confidence scores, and class
        predictions for each detected object.

        Args:
            input (Image): The input image on which to perform object detection.

        Returns:
            list[ImageCrop]: A list of ImageCrop objects representing the detected
                              objects in the input image.
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
                results: Results = (self.model(img_path, iou=0.6))[0]
        else:
            results: Results = (self.model(image.get_path()))[0]

        detections: list[Image] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        coordinates = results.boxes.xywhn.clone()
        names = [
            results.names[class_name.int().item()] for class_name in results.boxes.cls
        ]

        if len(names) == 0:
            # No objects detected, return empty list
            return detections
        for index in range(len(coordinates)):
            x, y, w, h = coordinates[index]
            image_crop = image.make_crop(
                x_center=float(x),
                y_center=float(y),
                width_relative=float(w),
                height_relative=float(h),
                orientation=image.orientation,
                name=names[index],
            )
            detections.append(image_crop)
        print(f"Found {len(detections)} objects in the image.")
        return detections

    def train(self, *images: ImageCrop, **kwargs) -> DetMetrics | None:
        self._init_model()
        log = kwargs.get("log", None)
        if log is None:
            raise ValueError(
                "Log directory must be specified in kwargs with key 'log'."
            )
        log = kwargs["base_folder"] / log
        log.mkdir(parents=True, exist_ok=True)
        print(f"Training files will be saved to: {log}")
        classes = kwargs.pop("classes", [])
        label_matrix, image_paths = self._prepare_assets(classes, log, *images)

        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance for training.")

        print(f"Training with model: {self.model.model_name}")

        cv_folds: int | None = kwargs.pop("cv_folds", None)

        if cv_folds:

            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

            mskf = MultilabelStratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=kwargs.get("seed", 42)
            )
            fold_count = 1
            results = None
            for train_index, val_index in mskf.split(image_paths, label_matrix):
                train = [image_paths[i] for i in train_index]
                val = [image_paths[i] for i in val_index]
                kwargs["config_file"] = self._prepare_yolo_config(
                    classes,
                    log,
                    train,
                    val,
                    *images,
                )
                kwargs["log"] = f"{log.name}_fold_{fold_count}"
                params = self._prepare_params(**kwargs)
                if kwargs.get("preview", False):
                    self._preview_assets(log, classes)
                with change_dir(kwargs["base_folder"]):
                    results: DetMetrics | None = self.model.train(**params)
                    if results is None:
                        raise Exception(
                            "[red]Training failed, no results returned.[/red]"
                        )
                fold_count += 1
                # Reload the original model for the next fold
                self.model = YOLO(self.original_model_path)
            return results
        else:
            kwargs["config_file"] = self._prepare_yolo_config(
                classes, log, [], [], *images
            )
            params = self._prepare_params(**kwargs)
            with change_dir(kwargs["base_folder"]):
                results: DetMetrics | None = self.model.train(**params)
            if results is None:
                raise Exception("[red]Training failed, no results returned.[/red]")
            validation_results = YOLO(results.save_dir / "weights" / "best.pt").val()
            return results

    def _preview_assets(self, log: Path, classes: list[str]) -> None:
        from drawyolo.draw import draw_box_on_image_with_labels

        preview_dir = log / "preview"
        preview_dir.mkdir(exist_ok=True)

        image_types = [".jpg", ".png", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"]

        for image_type in image_types:
            for image_file in log.glob(f"*{image_type}"):
                dest = preview_dir / image_file.name
                shutil.copy(image_file, dest)
                output_path = preview_dir / image_file.name
                draw_box_on_image_with_labels(
                    image=image_file,
                    output=output_path,
                    labels=image_file.with_suffix(".txt"),
                    classes=classes,
                )

    def _prepare_yolo_config(
        self,
        classes: list[str],
        log: Path,
        train: list[Path] = [],
        val: list[Path] = [],
        *images: ImageCrop,
    ) -> Path:

        train_files = (
            [f"./{Path(p).name}" for p in train]
            if train
            else list(
                set(
                    [
                        f"./{Path(img.get_path()).name}"
                        for img in images
                        if not img.validation
                    ]
                )
            )
        )

        val_files = (
            [f"./{Path(p).name}" for p in val]
            if val
            else list(
                set(
                    [
                        f"./{Path(img.get_path()).name}"
                        for img in images
                        if img.validation
                    ]
                )
            )
        )

        Path(log / "train.txt").write_text("\n".join(train_files))
        Path(log / "val.txt").write_text("\n".join(val_files))

        config = (
            f"train: train.txt\nval: val.txt\nnc: {len(classes)}\nnames: {classes}\n"
        )
        config_file = log / "config.yml"
        Path(config_file).write_text(config)

        print(
            f"Saved {len(train_files)} training images and {len(val_files)} validation images to {log}"
        )

        self._check_distribution(classes, train_files, val_files, *images)

        return config_file

    def _check_distribution(
        self,
        classes: list[str],
        train_files: list[str],
        val_files: list[str],
        *images: ImageCrop,
    ) -> None:
        train_files = [Path(f).name for f in train_files]
        val_files = [Path(f).name for f in val_files]
        train_classes = {name: 0 for name in classes}
        val_classes = {name: 0 for name in classes}
        for im in images:
            if im.get_path().name in val_files and im.name in classes:
                val_classes[im.name] += 1
            elif im.get_path().name in train_files and im.name in classes:
                train_classes[im.name] += 1

        table = Table(title="Class Distribution", show_lines=True)
        table.add_column(
            "Class Name", justify="left", style="green", header_style="bold green"
        )
        table.add_column(
            "Train Count", justify="right", style="red", header_style="bold red"
        )
        table.add_column(
            "Validation Count", justify="right", style="blue", header_style="bold blue"
        )

        for class_name in classes:
            table.add_row(
                class_name,
                str(train_classes[class_name]),
                str(val_classes[class_name]),
            )

        console = Console()
        console.print(table)

    def _write_yolo_label(self, name_index: int, img: ImageCrop, f):
        if name_index == -1:
            f.write("")
        else:
            f.write(
                f"{name_index} {img.x_center:.6f} {img.y_center:.6f} {img.width_relative:.6f} {img.height_relative:.6f}\n"
            )

    def _prepare_assets(
        self, classes: list[str], log: Path, *images: ImageCrop
    ) -> tuple[list[list[int]], list[str]]:
        image_paths = list(set([img.get_path().name for img in images]))
        label_matrix = [len(classes) * [0] for _ in range(len(image_paths))]
        for img in images:
            src = img.get_path()
            dst = log / src.name
            if isinstance(img, ImageCrop) and img.source_parent:
                dst = log / f"{src.stem}-{str(img.source_parent.id)}{src.suffix}"
            if not dst.exists():
                im = (
                    img.source_parent.pil()
                    if isinstance(img, ImageCrop) and img.source_parent
                    else img.pil()
                )
                im.save(dst)
            if isinstance(img, ImageCrop) and isinstance(img.source_parent, ImageCrop):
                img.set_rel_to_src_parent()
            name_index = classes.index(img.name) if img.name in classes else -1
            text_dst = log / f"{dst.stem}.txt"
            label_matrix[image_paths.index(img.get_path().name)][name_index] = 1
            write_mode = "a" if text_dst.exists() else "w"
            with open(text_dst, write_mode) as f:
                self._write_yolo_label(name_index, img, f)
        return label_matrix, image_paths

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
        }
        return params
