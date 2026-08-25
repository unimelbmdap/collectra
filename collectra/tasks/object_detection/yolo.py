from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from collectra.cli import command
from collectra.types.images import Image, ImageCrop
from collectra.types.texts import Text
from collectra.utils import change_dir

from collectra.utils import threading_locked
from ..machine_learning.training import (
    prepare_object_detection_inputs,
    print_distribution_table,
    run_training_command,
)
from ..machine_learning.yolo import YOLOTask

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO
    from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics

__all__ = ["ObjectDetectionYOLO"]


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


class ObjectDetectionYOLO(YOLOTask):
    singletons: bool = False

    @threading_locked()
    def run(self, *args: Image, **kwargs) -> list[Image]:
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
        from torchvision.ops import batched_nms

        if isinstance(image, ImageCrop):
            with tempfile.TemporaryDirectory() as temp_dir:
                img = image.pil()
                img_path = Path(temp_dir) / f"{Path(image.get_path()).stem}.png"
                img.save(img_path, format="PNG")
                results: Results = (self.model(img_path, iou=0.7, conf=0.25))[0]
        else:
            results: Results = (self.model(image.get_path(), iou=0.7, conf=0.25))[0]

        if getattr(self.model.model, "end2end", False) and len(results.boxes):
            keep = batched_nms(
                results.boxes.xyxy,
                results.boxes.conf,
                results.boxes.cls,
                iou_threshold=0.7,
            )
            results = results[keep]

        detections: list[Image] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        coordinates = results.boxes.xywhn.clone()
        names = [
            results.names[class_name.int().item()] for class_name in results.boxes.cls
        ]

        counts = Counter()
        if len(names) == 0:
            # No objects detected, return empty list
            return detections
        for index in range(len(coordinates)):
            name = names[index]

            if self.singletons and counts[name]:
                continue

            x, y, w, h = coordinates[index]
            confidence = results.boxes.conf[index]

            image_crop = image.make_crop(
                x_center=float(x),
                y_center=float(y),
                width_relative=float(w),
                height_relative=float(h),
                confidence=confidence.item(),
                orientation=image.orientation,
                name=name,
            )
            counts[name] += 1
            detections.append(image_crop)

        print(f"Found {len(detections)} objects in the image:")
        for name, count in counts.items():
            print(f"\t{name}: {count}")
        return detections

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
        imgsz: int = 640,
        early_stop: int = 50,
        preview: bool = False,
        workers: int = 8,
        pretrained: bool = True,
        optimizer: str = "auto",
        seed: int = 0,
        deterministic: bool = True,
        single_cls: bool = False,
        rect: bool = False,
        cos_lr: bool = False,
        close_mosaic: int = 10,
        amp: bool = True,
        fraction: float = 1.0,
        freeze: int | None = None,
        multi_scale: float = 0.0,
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: float = 3.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        box: float = 7.5,
        cls: float = 0.5,
        cls_pw: float = 0.0,
        dfl: float = 1.5,
        nbs: int = 64,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        flipud: float = 0.0,
        fliplr: float = 0.5,
        bgr: float = 0.0,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        cutmix: float = 0.0,
        time: float = 0.0,
        save: bool = True,
        save_period: int = -1,
        cache: str = "",
        device: str = "",
        verbose: bool = True,
        resume: bool = False,
        compile: str = "",
        distill_model: str = "",
        dis: float = 6.0,
        plots: bool = True,
    ):
        """Train this YOLO object detector."""
        return run_training_command(
            self,
            inputs,
            self._train,
            keep_log=keep_log,
            output=output,
            validation=validation,
            exclude=exclude,
            prepare_inputs=prepare_object_detection_inputs,
            include_unlabelled=True,
            model=model,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            early_stop=early_stop,
            preview=preview,
            workers=workers,
            pretrained=pretrained,
            optimizer=optimizer,
            seed=seed,
            deterministic=deterministic,
            single_cls=single_cls,
            rect=rect,
            cos_lr=cos_lr,
            close_mosaic=close_mosaic,
            amp=amp,
            fraction=fraction,
            freeze=freeze,
            multi_scale=multi_scale,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            box=box,
            cls=cls,
            cls_pw=cls_pw,
            dfl=dfl,
            nbs=nbs,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            bgr=bgr,
            mosaic=mosaic,
            mixup=mixup,
            cutmix=cutmix,
            time=time,
            save=save,
            save_period=save_period,
            cache=cache,
            device=device,
            verbose=verbose,
            resume=resume,
            compile=compile,
            distill_model=distill_model,
            dis=dis,
            plots=plots,
        )

    def _train(
        self, *images: ImageCrop, **kwargs
    ) -> DetMetrics | ClassifyMetrics | None:
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
        from ultralytics.models import YOLO

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

        print_distribution_table(
            "Class Distribution", metadata["classes"], train_classes, val_classes
        )

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
                if hasattr(img, "source_parent") and img.source_parent is not None:
                    img.source_parent.save(dst)
                else:
                    img.save(dst)
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
            "device": kwargs.get("device")
            or (
                "mps"
                if platform.system() == "Darwin"
                else "cuda" if torch.cuda.is_available() else "cpu"
            ),
            "epochs": kwargs.get("epochs", 100),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("early_stop", 50),
            "batch": kwargs.get("batch", 16),
            "workers": kwargs.get("workers", 8),
            "pretrained": kwargs.get("pretrained", True),
            "optimizer": kwargs.get("optimizer", "auto"),
            "seed": kwargs.get("seed", 0),
            "deterministic": kwargs.get("deterministic", True),
            "single_cls": kwargs.get("single_cls", False),
            "rect": kwargs.get("rect", False),
            "cos_lr": kwargs.get("cos_lr", False),
            "close_mosaic": kwargs.get("close_mosaic", 10),
            "amp": kwargs.get("amp", True),
            "fraction": kwargs.get("fraction", 1.0),
            "multi_scale": kwargs.get("multi_scale", 0.0),
            "lr0": kwargs.get("lr0", 0.01),
            "lrf": kwargs.get("lrf", 0.01),
            "momentum": kwargs.get("momentum", 0.937),
            "weight_decay": kwargs.get("weight_decay", 0.0005),
            "warmup_epochs": kwargs.get("warmup_epochs", 3.0),
            "warmup_momentum": kwargs.get("warmup_momentum", 0.8),
            "warmup_bias_lr": kwargs.get("warmup_bias_lr", 0.1),
            "box": kwargs.get("box", 7.5),
            "cls": kwargs.get("cls", 0.5),
            "cls_pw": kwargs.get("cls_pw", 0.0),
            "dfl": kwargs.get("dfl", 1.5),
            "nbs": kwargs.get("nbs", 64),
            "hsv_h": kwargs.get("hsv_h", 0.015),
            "hsv_s": kwargs.get("hsv_s", 0.7),
            "hsv_v": kwargs.get("hsv_v", 0.4),
            "degrees": kwargs.get("degrees", 0.0),
            "translate": kwargs.get("translate", 0.1),
            "scale": kwargs.get("scale", 0.5),
            "shear": kwargs.get("shear", 0.0),
            "perspective": kwargs.get("perspective", 0.0),
            "flipud": kwargs.get("flipud", 0.0),
            "fliplr": kwargs.get("fliplr", 0.5),
            "bgr": kwargs.get("bgr", 0.0),
            "mosaic": kwargs.get("mosaic", 1.0),
            "mixup": kwargs.get("mixup", 0.0),
            "cutmix": kwargs.get("cutmix", 0.0),
            "save": kwargs.get("save", True),
            "save_period": kwargs.get("save_period", -1),
            "verbose": kwargs.get("verbose", True),
            "resume": kwargs.get("resume", False),
            "dis": kwargs.get("dis", 6.0),
            "plots": kwargs.get("plots", True),
        }
        if kwargs.get("freeze") is not None:
            params["freeze"] = kwargs["freeze"]
        for optional in ("time", "cache", "compile", "distill_model"):
            if kwargs.get(optional):
                params[optional] = kwargs[optional]
        return params
