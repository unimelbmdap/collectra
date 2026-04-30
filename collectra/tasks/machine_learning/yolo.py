import shutil
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table
from ultralytics.engine.results import Results
from ultralytics.models import YOLO
from ultralytics.utils import ThreadingLocked
from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics

from collectra.types.images import Image, ImageCrop
from collectra.types.texts import Text
from collectra.utils import change_dir

from ...logger import get_logger
from .base import MachineLearningTask

logger = get_logger(__name__)

__all__ = ["ObjectDetectionYOLO", "ClassifierYOLO"]


class DetectionResult:
    metrics: dict
    best_result: DetMetrics

    def __init__(self, result: DetMetrics) -> None:
        self.best_result = result
        self.metrics = result.results_dict if result else {}

    @staticmethod
    def get_best_result(
        results: list["DetectionResult"], metric: str = "metrics/mAP50-95(B)"
    ) -> DetMetrics:
        best_result: DetMetrics | None = None
        best_metric_result = 0.0
        for result in results:
            if result is None:
                continue
            metric_result = result.metrics.get(metric, 0.0)
            if best_result is None:
                best_metric_result = metric_result
                best_result = result.best_result
                continue
            if metric_result > best_metric_result:
                best_metric_result = metric_result
                best_result = result.best_result
        if best_result is None:
            raise ValueError("No valid results found to determine the best result.")
        return best_result


class ObjectDetectionYOLO(MachineLearningTask):

    model: str | Path | YOLO
    original_model_path: str | Path = ""

    def _init_model(self) -> None:
        """Ensure that the YOLO model is loaded before performing any operations."""
        self._load()
        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance")

    def _reload(self) -> None:
        """Reload the YOLO model from the original model path."""
        if not self.original_model_path:
            logger.warning("Original model path is not set. Skipping...")
        self.model = self.original_model_path
        self._load()

    def _load(self) -> None:
        """Load the YOLO model for object detection.

        Args:
            model (str | Path): The path to the model file or the model itself.
        """
        if self.model and isinstance(self.model, (str, Path)):
            self.original_model_path = self.model
            self.model = YOLO(Path(self.model))
            return
        logger.warning("Model is already loaded or invalid model path provided.")

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
                img_path = Path(temp_dir) / f"{Path(image.get_path()).stem}.png"
                img.save(img_path, format="PNG")
                results: Results = (self.model(img_path, iou=0.8))[0]
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

    def train(
        self, *images: ImageCrop, **kwargs
    ) -> DetMetrics | ClassifyMetrics | None:
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

        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance for training.")

        print(f"Training with model: {self.model.model_name}")

        validation = kwargs.get("validation", "")
        exclude = kwargs.get("exclude", "")
        train, val = self._prepare_assets(classes, log, validation, exclude, *images)
        return self._train_fold(train, val, classes, log, kwargs)

    def _train_fold(
        self,
        train: list[str] | Path,
        val: list[str] | Path,
        classes: list[str],
        log: Path,
        kwargs: dict,
        fold_count: int | None = None,
    ) -> DetMetrics | ClassifyMetrics:
        if not isinstance(self.model, YOLO):
            raise ValueError("Expected model to be a YOLO instance for training.")
        kwargs["config_file"] = self._prepare_yolo_config(
            log,
            classes,
            train,
            val,
        )
        kwargs["log"] = f"{log.name}"
        if fold_count:
            kwargs["log"] += f"_fold_{fold_count}"
        params = self._prepare_params(**kwargs)
        if kwargs.get("preview", False):
            self._preview_assets(log, classes)
        with change_dir(kwargs["base_folder"]):
            results: DetMetrics | None = self.model.train(**params)
        if results is None:
            raise Exception("[red]Training failed, no results returned.[/red]")
        self._reload()
        return results

    def _preview_assets(self, log: Path, classes: list[str]) -> None:
        from drawyolo.draw import draw_box_on_image_with_labels

        preview_dir = log / "preview"
        preview_dir.mkdir(exist_ok=True)

        for image_type in Image.image_types():
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
        log: Path,
        classes: list[str],
        train: list[str] | Path,
        val: list[str] | Path,
    ) -> Path:

        metadata = {
            "classes": classes,
            "train": train,
            "val": val,
        }

        Path(log / "train.txt").write_text(
            "\n".join([f"./{Path(p).name}" for p in train])
        )
        Path(log / "val.txt").write_text("\n".join([f"./{Path(p).name}" for p in val]))

        print(
            f"Saved {len(metadata['train'])} training images and {len(metadata['val'])} validation images to {log}"
        )

        self._check_distribution(log, metadata)

        train_txt = "train.txt"
        val_txt = "val.txt"
        num_classes = len(metadata["classes"])
        names = metadata["classes"]

        config = (
            f"train: {train_txt}\nval: {val_txt}\nnc: {num_classes}\nnames: {names}\n"
        )
        config_file = log / "config.yml"
        Path(config_file).write_text(config)

        return config_file

    def _check_distribution(
        self,
        log: Path,
        metadata: dict | list[str],
    ) -> None:
        if not isinstance(metadata, dict):
            return

        train_classes = {name: 0 for name in metadata["classes"]}
        val_classes = {name: 0 for name in metadata["classes"]}

        for image_type in Image.image_types():
            for image_file in log.glob(f"*{image_type}"):
                label_file = image_file.with_suffix(".txt")
                with open(label_file, "r") as f:
                    label_lines = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                for line in label_lines:
                    class_index = int(line.split()[0])
                    class_name = metadata["classes"][class_index]
                    if image_file.name in metadata["train"]:
                        train_classes[class_name] += 1
                    elif image_file.name in metadata["val"]:
                        val_classes[class_name] += 1

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

        for class_name in metadata["classes"]:
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
        self,
        classes: list[str],
        log: Path,
        validation_flag: str,
        exclude_flag: str,
        *images: ImageCrop,
    ) -> tuple[list[str], list[str]] | tuple[Path, Path]:
        """Prepare assets for YOLO training.

        Args:
            classes (list[str]): List of class names.
            log (Path): Path to store the assets for training.
            validation_flag (str): The value of the validation flag to identify validation images.
            exclude_flag (str): The value of the exclude flag to identify images to be excluded from training.
            images (ImageCrop): Variable number of ImageCrop instances to be prepared.

        Returns:
            tuple[list[list[int]], list[str], list[str]]: A tuple containing the label matrix for iterative stratification,
            training image filenames, and validation image filenames.

        """
        train: list[str] = list()
        val: list[str] = list()
        for img in images:
            src = img.get_path()
            dst = log / src.name
            if isinstance(img, ImageCrop) and img.source_parent:
                dst = log / f"{src.stem}-{str(img.source_parent.id)}{src.suffix}"
            if not dst.exists():
                if not hasattr(img, "source_parent"):
                    img.save(dst)
                else:
                    img.source_parent.save(dst)
                if img.partition == validation_flag:
                    val.append(dst.name)
                elif not exclude_flag or img.partition != exclude_flag:
                    train.append(dst.name)
            if isinstance(img, ImageCrop) and isinstance(img.source_parent, ImageCrop):
                img.set_rel_to_src_parent()
            name_index = classes.index(img.name) if img.name in classes else -1
            text_dst = log / f"{dst.stem}.txt"
            write_mode = "a" if text_dst.exists() else "w"
            with open(text_dst, write_mode) as f:
                self._write_yolo_label(name_index, img, f)
        return train, val

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


class ClassifierYOLO(ObjectDetectionYOLO):

    @ThreadingLocked()
    def run(self, *args: Image) -> Text:
        """Run image classification inference on the provided Image.

        Args:
            *args: A single Image (or ImageCrop) to classify.

        Returns:
            Text: A Text object with the predicted class name.
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
        print(f"Classified as: {predicted_class}")

        return Text(name=self.get_output_name(), data=predicted_class)

    def train(self, *images: Image, **kwargs) -> ClassifyMetrics | None:
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
