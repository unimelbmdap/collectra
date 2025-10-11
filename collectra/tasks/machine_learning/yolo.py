import shutil

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
            self._load(self.model)
        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance")

    def _load(self, model: str | Path) -> None:
        """Load the YOLO model for object detection.

        Args:
            model (str | Path): The path to the model file or the model itself.
        """
        self.model = YOLO(model)

    @ThreadingLocked()
    def run(self, input: Image) -> list[ImageCrop]:
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
        self._init_model()
        results: Results = (self.model(input.get_path())).pop()
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
                path=input.get_path(),
                x_center=float(x),
                y_center=float(y),
                width_relative=float(w),
                height_relative=float(h),
            )
            detections.append(image_crop)
        print(f"Found {len(detections)} objects in the image.")
        return detections

    def train(
        self,
        train_img: list[ImageCrop],
        val_img: list[ImageCrop],
        classes: list[str],
        **kwargs,
    ) -> DetMetrics | None:
        self._init_model()
        log_dir: Path = (
            Path(kwargs.pop("log_dir"))
            if kwargs.get("log_dir", None)
            else Path.cwd() / "logs"
        )
        kwargs["log_dir"] = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training logs will be saved to: {log_dir}")
        kwargs["config_file"] = self._prepare_yolo_config(
            classes, train_img, val_img, log_dir
        )
        self._prepare_assets(classes, train_img + val_img, log_dir)
        params = self._prepare_params(**kwargs)
        results: DetMetrics | None = self.model.train(**params)
        return results

    def _prepare_yolo_config(
        self,
        classes: list[str],
        train_img: list[ImageCrop],
        val_img: list[ImageCrop],
        log_dir: Path,
    ) -> Path:
        config = (
            f"train: train.txt\nval: val.txt\nnc: {len(classes)}\nnames: {classes}\n"
        )
        train_files = set([f"./{img.get_path().name}" for img in train_img])
        val_files = set([f"./{img.get_path().name}" for img in val_img])

        config_file = log_dir / "config.yml"
        Path(config_file).write_text(config)
        if train_files:
            Path(log_dir / "train.txt").write_text("\n".join(train_files))
        if val_files:
            Path(log_dir / "val.txt").write_text("\n".join(val_files))
        return config_file

    def _prepare_assets(
        self, classes: list[str], images: list[ImageCrop], log_dir: Path
    ) -> None:
        for img in images:
            src = img.get_path()
            dst = log_dir / src.name
            if not dst.exists():  # Only copy if the file does not already exist
                shutil.copy(src, dst)
            name_index = classes.index(img.name) if img.name in classes else -1
            if not name_index:
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
            "data": kwargs["config_file"],
            "project": kwargs["config_file"].parent,
            "device": (
                "mps"
                if platform.system() == "Darwin"
                else "cuda" if torch.cuda.is_available() else "cpu"
            ),
            "epochs": kwargs.get("epochs", 1),
            "imgsz": kwargs.get("imgsz", 640),
        }
        return params
