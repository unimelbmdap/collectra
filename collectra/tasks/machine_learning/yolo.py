import shutil

from typing import overload
from ultralytics.models import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.metrics import DetMetrics

from pathlib import Path

from collectra.types.images import Image, ImageCrop
from ultralytics.utils import ThreadingLocked

from .base import MachineLearningTask

__all__ = ["ObjectDetectionYOLO"]


class ObjectDetectionYOLO(MachineLearningTask):

    model: str | Path | YOLO

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
        self.model = YOLO(model)

    @property
    def input_nums(self) -> int:
        return 1

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
        results: Results = (self.model(image.get_path())).pop()
        detections: list[ImageCrop] = []
        coordinates = results.boxes.xywhn if results.boxes else []
        names = (
            [results.names[cls.item()] for cls in results.boxes.cls.int()]
            if results.boxes
            else []
        )
        for index in range(len(coordinates)):
            x, y, w, h = coordinates[index]
            image_crop = ImageCrop(
                name=names[index],
                data=image.get_path(),
                x_center=float(x),
                y_center=float(y),
                width_relative=float(w),
                height_relative=float(h),
            )
            detections.append(image_crop)
        print(f"Found {len(detections)} objects in the image.")
        return detections

    def train(
        self, *images: ImageCrop, **kwargs
    ) -> tuple[DetMetrics | None, DetMetrics | None]:
        self._init_model()
        log_dir: Path = (
            Path(kwargs.pop("log_dir"))
            if kwargs.get("log_dir", None)
            else Path.cwd() / "log_dir"
        )
        kwargs["log_dir"] = log_dir
        classes = kwargs.pop("classes", [])
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training logs will be saved to: {log_dir}")
        kwargs["config_file"] = self._prepare_yolo_config(classes, log_dir, *images)
        self._prepare_assets(classes, log_dir, *images)
        params = self._prepare_params(**kwargs)
        results: DetMetrics | None = self.model.train(**params)
        validation_results = YOLO(results.save_dir / "weights" / "best.pt").val()
        return results, validation_results

    def _prepare_yolo_config(
        self, classes: list[str], log_dir: Path, *images: ImageCrop
    ) -> Path:
        config = (
            f"train: train.txt\nval: val.txt\nnc: {len(classes)}\nnames: {classes}\n"
        )
        train_files = set()
        val_files = set()
        for img in images:
            if img.validation:
                val_files.add(f"./{Path(img.get_path()).name}")
            else:
                train_files.add(f"./{Path(img.get_path()).name}")
        config_file = log_dir / "config.yml"
        Path(config_file).write_text(config)
        Path(log_dir / "train.txt").write_text("\n".join(train_files))
        Path(log_dir / "val.txt").write_text("\n".join(val_files))
        return config_file

    def _prepare_assets(
        self, classes: list[str], log_dir: Path, *images: ImageCrop
    ) -> None:
        for img in images:
            src = Path(img.get_path())
            dst = log_dir / src.name
            if not dst.exists():  # Only copy if the file does not already exist
                shutil.copy(src, dst)
            name_index = classes.index(img.name) if img.name in classes else -1
            if name_index == -1:
                continue
            text_dst = log_dir / f"{src.stem}.txt"
            if not text_dst.exists():
                with open(text_dst, "w") as f:
                    f.write(
                        f"{name_index} {img.x_center:.6f} {img.y_center:.6f} {img.width_relative:.6f} {img.height_relative:.6f}\n"
                    )
            else:
                with open(text_dst, "a") as f:
                    f.write(
                        f"{name_index} {img.x_center:.6f} {img.y_center:.6f} {img.width_relative:.6f} {img.height_relative:.6f}\n"
                    )

    def _prepare_params(self, **kwargs) -> dict:
        import platform, torch        
        params = {
            "name": kwargs["config_file"].parent,
            "data": kwargs["config_file"],
            "project": kwargs["project"],
            "device": (
                "mps"
                if platform.system() == "Darwin"
                else "cuda" if torch.cuda.is_available() else "cpu"
            ),
            "epochs": kwargs.get("epochs", 1),
            "imgsz": kwargs.get("imgsz", 640),            
        }        
        return params
