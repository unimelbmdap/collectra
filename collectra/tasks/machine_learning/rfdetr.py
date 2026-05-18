import json
import shutil
from pathlib import Path

from rfdetr import RFDETRBase
from rich.console import Console
from rich.table import Table
from ultralytics.utils import ThreadingLocked

from collectra.types.images import Image, ImageCrop
from collectra.utils import change_dir

from ...logger import get_logger
from .base import MachineLearningTask
from .detr import DetectionTrainResult

logger = get_logger(__name__)

__all__ = ["ObjectDetectionRFDETR"]


class ObjectDetectionRFDETR(MachineLearningTask):
    """RF-DETR task backed by the Roboflow rfdetr package."""

    model: str | Path | object | None
    original_model_path: str | Path | None = None

    def __init__(
        self,
        name: str,
        model: str | Path | object | None = None,
        **kwargs,
    ):
        super().__init__(name, model=model, **kwargs)
        self._device = None
        self._categories: list[str] = []

    def _select_device(self) -> str:
        import platform

        import torch

        return (
            "mps"
            if platform.system() == "Darwin"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _init_model(self) -> None:
        self._load()

        if not isinstance(self.model, RFDETRBase):
            raise ValueError("Model must be an RFDETRBase instance")

    def _reload(self) -> None:
        if not self.original_model_path:
            logger.warning("Original model path is not set. Skipping...")
            return

        self.model = RFDETRBase(pretrain_weights=str(self.original_model_path))
        self._load_categories_from_json(Path(self.original_model_path))

    def _classes_sidecar_candidates(self, checkpoint_path: Path) -> list[Path]:
        weights_dir = checkpoint_path.parent
        return [
            weights_dir / f"{checkpoint_path.stem}.classes.json",
            weights_dir / "classes.json",
        ]

    def _load_categories_from_json(self, checkpoint_path: Path) -> None:
        for classes_file in self._classes_sidecar_candidates(checkpoint_path):
            if classes_file.exists():
                with open(classes_file, "r") as f:
                    metadata = json.load(f)
                self._categories = metadata.get("classes", self._categories)
                return

    def _load(self) -> None:
        if isinstance(self.model, RFDETRBase):
            return

        self._device = self._select_device()

        if self.model is None or (
            isinstance(self.model, str) and self.model == "default"
        ):
            self.model = RFDETRBase()
            from rfdetr.assets.coco_classes import COCO_CLASSES

            self._categories = list(COCO_CLASSES)
            return

        if not isinstance(self.model, (str, Path)):
            raise ValueError(
                "Model must be None, 'default', a checkpoint path, or an RFDETRBase instance"
            )

        checkpoint_path = Path(str(self.model))
        if checkpoint_path.exists() and checkpoint_path.is_file():
            self.original_model_path = checkpoint_path
            self.model = RFDETRBase(pretrain_weights=str(checkpoint_path))
            self._load_categories_from_json(checkpoint_path)
            candidates = self._classes_sidecar_candidates(checkpoint_path)
            if not any(c.exists() for c in candidates):
                logger.warning(
                    "No classes sidecar found alongside checkpoint %s "
                    "(looked for %s); inference labels will fall back to "
                    "numeric class indices because no class metadata is available.",
                    checkpoint_path,
                    [str(c) for c in candidates],
                )
            return

        raise ValueError(
            f"Model path does not exist: {checkpoint_path}. "
            "RF-DETR requires None/'default' for pretrained or a valid checkpoint path."
        )

    @ThreadingLocked()
    def run(self, *args: Image) -> list[Image]:
        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        if not isinstance(args[0], Image):
            raise TypeError("Input must be an instance of Image.")

        image = args[0]
        self._init_model()

        threshold = float(getattr(self, "threshold", 0.5))
        pil_image = image.pil().convert("RGB")
        width, height = pil_image.size

        detections = self.model.predict(pil_image, threshold=threshold)

        results: list[Image] = []
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            x1 = max(0.0, min(float(width), float(x1)))
            x2 = max(0.0, min(float(width), float(x2)))
            y1 = max(0.0, min(float(height), float(y1)))
            y2 = max(0.0, min(float(height), float(y2)))

            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            x_center = (x1 + x2) / 2.0 / width
            y_center = (y1 + y2) / 2.0 / height
            width_relative = bbox_width / width
            height_relative = bbox_height / height

            class_idx = int(detections.class_id[i])
            class_name = (
                self._categories[class_idx]
                if 0 <= class_idx < len(self._categories)
                else str(class_idx)
            )

            results.append(
                image.make_crop(
                    x_center=x_center,
                    y_center=y_center,
                    width_relative=width_relative,
                    height_relative=height_relative,
                    orientation=image.orientation,
                    name=class_name,
                )
            )

        print(f"Found {len(results)} objects in the image.")
        return results

    def _prepare_params(self, **kwargs) -> dict:
        return {
            "epochs": int(kwargs.get("epochs", 1)),
            "batch_size": int(kwargs.get("batch", 4)),
            "grad_accum_steps": int(kwargs.get("grad_accum_steps", 4)),
            "lr": float(kwargs.get("lr", 1e-4)),
            "output_dir": str(kwargs["output_dir"]),
            "wandb": bool(kwargs.get("wandb", True)),
            "project": kwargs.get("project", "runs/rfdetr"),
            "run": kwargs.get("log", "rfdetr-run"),
            "early_stopping": bool(kwargs.get("early_stopping", False)),
        }

    def _check_distribution(
        self,
        classes: list[str],
        train_samples: list[dict],
        val_samples: list[dict],
    ) -> None:
        train_classes = {name: 0 for name in classes}
        val_classes = {name: 0 for name in classes}
        label_to_name = {index: name for index, name in enumerate(classes)}

        for sample in train_samples:
            for label in sample["labels"]:
                class_name = label_to_name.get(label, "")
                if class_name:
                    train_classes[class_name] += 1

        for sample in val_samples:
            for label in sample["labels"]:
                class_name = label_to_name.get(label, "")
                if class_name:
                    val_classes[class_name] += 1

        table = Table(title="Class Distribution", show_lines=True)
        table.add_column(
            "Class Name", justify="left", style="green", header_style="bold green"
        )
        table.add_column(
            "Train Count", justify="right", style="red", header_style="bold red"
        )
        table.add_column(
            "Validation Count",
            justify="right",
            style="blue",
            header_style="bold blue",
        )

        for class_name in classes:
            table.add_row(
                class_name,
                str(train_classes[class_name]),
                str(val_classes[class_name]),
            )

        Console().print(table)

    def _prepare_assets(
        self,
        classes: list[str],
        validation_flag: str,
        exclude_flag: str,
        *images: ImageCrop,
    ) -> tuple[list[dict], list[dict]]:
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        grouped: dict[Path, dict] = {}

        for img in images:
            if exclude_flag and img.partition == exclude_flag:
                continue

            base_img: Image | ImageCrop = img
            if isinstance(img, ImageCrop):
                if img.source_parent:
                    base_img = img.source_parent
                    if isinstance(img.source_parent, ImageCrop):
                        img.set_rel_to_src_parent()

            image_path = base_img.get_path()
            if image_path not in grouped:
                grouped[image_path] = {
                    "image_path": image_path,
                    "partition": img.partition,
                    "boxes": [],
                    "labels": [],
                }

            if not isinstance(img, ImageCrop):
                continue
            if img.name not in class_to_idx:
                continue

            width, height = base_img.width, base_img.height
            box_w = float(img.width_relative) * width
            box_h = float(img.height_relative) * height
            cx = float(img.x_center) * width
            cy = float(img.y_center) * height

            x1 = max(0.0, cx - box_w / 2.0)
            y1 = max(0.0, cy - box_h / 2.0)
            x2 = min(float(width), cx + box_w / 2.0)
            y2 = min(float(height), cy + box_h / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue

            grouped[image_path]["boxes"].append([x1, y1, x2, y2])
            grouped[image_path]["labels"].append(class_to_idx[img.name])

        train_samples: list[dict] = []
        val_samples: list[dict] = []
        for sample in grouped.values():
            if validation_flag and sample["partition"] == validation_flag:
                val_samples.append(sample)
            elif not exclude_flag or sample["partition"] != exclude_flag:
                train_samples.append(sample)
        return train_samples, val_samples

    def _write_coco_dataset(
        self,
        train_samples: list[dict],
        val_samples: list[dict],
        classes: list[str],
        dataset_dir: Path,
    ) -> Path:
        from PIL import Image as PILImage

        categories = [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(classes)
        ]

        for split_name, samples in [("train", train_samples), ("valid", val_samples)]:
            split_dir = dataset_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            images_list: list[dict] = []
            annotations_list: list[dict] = []
            annotation_id = 0

            for image_id, sample in enumerate(samples):
                src_path = Path(sample["image_path"])
                dest_name = f"{image_id}_{src_path.name}"
                dest_path = split_dir / dest_name
                shutil.copy2(src_path, dest_path)

                with PILImage.open(dest_path) as pil_img:
                    w, h = pil_img.size

                images_list.append(
                    {
                        "id": image_id,
                        "file_name": dest_name,
                        "width": w,
                        "height": h,
                    }
                )

                for box, label in zip(sample["boxes"], sample["labels"]):
                    x1, y1, x2, y2 = box
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                    area = (x2 - x1) * (y2 - y1)

                    annotations_list.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": coco_bbox,
                            "area": float(area),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

            coco_json = {
                "images": images_list,
                "annotations": annotations_list,
                "categories": categories,
            }

            annotations_path = split_dir / "_annotations.coco.json"
            with open(annotations_path, "w") as f:
                json.dump(coco_json, f, indent=2)

        return dataset_dir

    def _train_fold(
        self,
        train_samples: list[dict],
        val_samples: list[dict],
        classes: list[str],
        log: Path,
        kwargs: dict,
    ) -> DetectionTrainResult:

        weights_dir = log / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = log / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._write_coco_dataset(train_samples, val_samples, classes, dataset_dir)

        params_kwargs = dict(kwargs)
        params_kwargs["log"] = f"{log.name}"
        params_kwargs["output_dir"] = str(weights_dir)
        params = self._prepare_params(**params_kwargs)

        model = self.model
        model.train(
            dataset_dir=str(dataset_dir),
            **params,
        )

        self._categories = classes
        self.model = model

        # Persist class names alongside checkpoint for later reloading
        classes_metadata = {"classes": classes}
        with open(weights_dir / "classes.json", "w") as f:
            json.dump(classes_metadata, f, indent=2)

        # RF-DETR saves checkpoints as checkpoint_best_regular.pth
        # Copy to best.pt for consistency with DETR/YOLO pattern
        best_pt = weights_dir / "best.pt"
        for candidate in [
            weights_dir / "checkpoint_best_regular.pth",
            weights_dir / "checkpoint_best_total.pth",
        ]:
            if candidate.exists():
                shutil.copy2(candidate, best_pt)
                break

        if best_pt.exists():
            self.original_model_path = best_pt
            self._reload()

        best_metrics = {"status": "completed", "classes": classes}

        logger.info("RF-DETR training completed. Weights saved to: %s", weights_dir)
        return DetectionTrainResult(save_dir=log, results_dict=best_metrics)

    def train(self, *images: ImageCrop, **kwargs) -> DetectionTrainResult:
        self._init_model()
        log = kwargs.get("log", None)
        if log is None:
            raise ValueError(
                "Log directory must be specified in kwargs with key 'log'."
            )
        if "base_folder" not in kwargs:
            raise ValueError(
                "Base folder must be specified in kwargs with key 'base_folder'."
            )

        log = kwargs["base_folder"] / log
        log.mkdir(parents=True, exist_ok=True)
        print(f"Training files will be saved to: {log}")

        classes = kwargs.get("classes", [])
        if not classes:
            classes = sorted({img.name for img in images if isinstance(img, ImageCrop)})
        if not classes:
            raise ValueError("No classes provided/found for RF-DETR training.")

        print("Training with model: RF-DETR")

        validation = kwargs.get("validation", "")
        exclude = kwargs.get("exclude", "")
        train_samples, val_samples = self._prepare_assets(
            classes,
            validation,
            exclude,
            *images,
        )
        self._check_distribution(classes, train_samples, val_samples)

        with change_dir(kwargs["base_folder"]):
            results = self._train_fold(train_samples, val_samples, classes, log, kwargs)

        print(
            f"Best RF-DETR model saved to: {results.save_dir / 'weights' / 'best.pt'}"
        )
        return results
