"""YOLO instance segmentation with Collectra's shared mask training format."""

from __future__ import annotations

import json
import shutil
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from cappa import Arg
import numpy as np
from PIL import Image as PILImage

from collectra.cli import command
from collectra.tasks.base import Task
from collectra.types.images import Image, ImageSegmentation
from collectra.utils import threading_locked

from .data import SegmentationTrainingData, mask_to_source


from collectra.tasks.machine_learning.yolo import get_yolo_class


def yolo_device(device, num_gpus=1):
    """Translate torch-style devices to Ultralytics device selectors."""
    if device == "cuda" or str(device).startswith("cuda:"):
        start = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        return list(range(start, start + num_gpus)) if num_gpus > 1 else start
    if num_gpus != 1:
        raise ValueError("Multiple GPUs require a CUDA device")
    return device


class ImageSegmentationYOLO(SegmentationTrainingData, Task):
    """Predict one source-aligned ImageSegmentation for each detected instance.

    ``model`` is an Ultralytics segmentation model name or a local YOLO
    checkpoint. Training learns the annotation names as classes. SAM prompt and
    version settings are accepted for pipeline migration, but are not used.
    """

    def __init__(
        self, name: str, model: str | Path = "yolo11n-seg.pt", threshold: float = 0.5,
        device: str = "cuda", resolution: int | None = None, **kwargs,
    ):
        super().__init__(name, **kwargs)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if resolution is not None and resolution < 1:
            raise ValueError("resolution must be positive")
        self.model, self.threshold, self.device = model or "yolo11n-seg.pt", threshold, device
        self.resolution = resolution
        self._predictor = None

    def attributes_to_ignore(self):
        return super().attributes_to_ignore() | {"_predictor"}

    def input_type(self) -> type:
        return Image

    def output_type(self) -> type:
        return ImageSegmentation

    @staticmethod
    def _progress(message):
        print(f"[YOLO Seg] {message}", flush=True)

    def _label_prompts(self, samples):
        # Preserve artefact names verbatim rather than SAM's prompt descriptions.
        return {item.name: item.name for sample in samples for item in sample.instances}

    def _model_path(self):
        path = Path(self.model).expanduser()
        if not path.is_absolute() and self.pipeline is not None:
            candidate = self.pipeline.path / path
            if candidate.is_file():
                return str(candidate)
        if path.is_absolute() or path.parent != Path("."):
            if not path.is_file():
                raise FileNotFoundError(path)
        return str(path)

    def _init_model(self):
        if self._predictor is None:
            predictor = get_yolo_class()(self._model_path())
            if predictor.task != "segment":
                raise ValueError("ImageSegmentationYOLO requires a YOLO segmentation model (-seg)")
            self._predictor = predictor

    @threading_locked()
    def run(self, *images: Image) -> list[ImageSegmentation]:
        if not images or any(not isinstance(image, Image) for image in images):
            raise TypeError("YOLO segmentation requires one or more Image artefacts")
        self._init_model()
        directory = Path(self.context.file_path or Path.cwd()) / "masks"
        directory.mkdir(parents=True, exist_ok=True)
        outputs = getattr(self, "output", None)
        outputs = [outputs] if isinstance(outputs, str) else outputs
        results = []
        for image in images:
            view = image.pil().convert("RGB")
            options = dict(conf=self.threshold, device=yolo_device(self.device), retina_masks=True, verbose=False)
            if self.resolution is not None:
                options["imgsz"] = self.resolution
            prediction, = self._predictor.predict(view, **options)
            if prediction.masks is None:
                continue
            masks = prediction.masks.data.detach().cpu().numpy()
            classes = prediction.boxes.cls.detach().cpu().numpy().astype(int)
            if masks.shape != (len(classes), view.height, view.width):
                raise ValueError("YOLO returned masks that do not match the input image view")
            for mask, class_id in zip(masks, classes):
                label = prediction.names[int(class_id)]
                if outputs:
                    aliases = {item.replace("_", " "): item for item in outputs}
                    aliases.update({prompt: item for item, prompt in getattr(self, "prompts", {}).items() if item in outputs})
                    label = aliases.get(label, label)
                    if len(outputs) == 1 and len(prediction.names) == 1:
                        label = outputs[0]
                if outputs and label not in outputs:
                    continue
                pixels = mask_to_source(image, mask > .5)
                if not pixels.any():
                    continue
                path = directory / f"yolo-{uuid.uuid4().hex}.png"
                PILImage.fromarray(pixels).save(path)
                results.append(ImageSegmentation(
                    name=label, data=image.get_path(), mask=path, parents=[image.id],
                    orientation=image.orientation, partition=image.partition,
                ))
        return results

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                required=False,
                default=[],
                help="Results YAML files, result directories, or parent directories containing annotated images and masks. May be omitted when reusing an existing export.",
            ),
        ],
        output: Annotated[
            Path | None,
            Arg(help="Directory for prepared data, metrics.csv, logs and best model weights. Defaults to a timestamped directory; required with --reuse-data."),
        ] = None,
        validation: Annotated[
            str,
            Arg(help="Partition used for validation and best-model selection by mask IoU. Omit to train without validation. Ignored when reusing an existing export."),
        ] = "",
        exclude: Annotated[
            str,
            Arg(help="Partition excluded from both training and validation during data preparation. Ignored when reusing an existing export."),
        ] = "",
        epochs: Annotated[
            int, Arg(help="Number of training epochs; validation runs after every epoch when available."),
        ] = 20,
        batch: Annotated[
            int, Arg(help="Number of training images per batch on each GPU."),
        ] = 1,
        num_gpus: Annotated[
            int, Arg(help="Number of GPUs used for training on this node."),
        ] = 1,
        keep_log: Annotated[
            bool,
            Arg(help="Keep prepared data, metrics, logs and checkpoints after copying the best weights into the pipeline. --no-keep-log removes the output directory."),
        ] = True,
        reuse_data: Annotated[
            bool,
            Arg(help="Reuse OUTPUT/dataset when both split annotation files exist; otherwise prepare it from the input paths. Requires --output. Reuse ignores input paths, partitions and --max-items. Does not resume optimizer state."),
        ] = False,
        wandb: Annotated[
            bool,
            Arg(help="Log epoch metrics to Weights & Biases. Project: <pipeline-directory>-<task>; run: output directory name; entity: W&B settings or WANDB_ENTITY."),
        ] = False,
        max_items: Annotated[
            int | None,
            Arg(help="Maximum total images across training and validation, after exclusions, in input order. Omit for all images. Ignored when reusing an existing export."),
        ] = None,
    ) -> Path:
        """Fine-tune YOLO on mask annotations and adopt its best weights."""
        if min(epochs, batch, num_gpus) < 1:
            raise ValueError("epochs, batch and num_gpus must be positive")
        if self.pipeline is None:
            raise RuntimeError("Attach the task to a pipeline before training")
        if reuse_data and output is None:
            raise ValueError("--reuse-data requires --output")
        output = (Path(output).expanduser().resolve() if output is not None
                  else Path.cwd() / datetime.now().strftime("%Y%m%d_%H%M%S"))
        dataset = output / "dataset"
        reuse_prepared = reuse_data and all(
            (dataset / split / "_annotations.coco.json").is_file()
            for split in ("train", "valid")
        )
        if not reuse_prepared:
            if max_items is not None and max_items < 1:
                raise ValueError("max_items must be positive")
            if not inputs:
                raise ValueError("Supply training inputs or reuse an existing OUTPUT/dataset export")
        output.mkdir(parents=True, exist_ok=True)
        self._progress(f"Training {self.name}: output={output}, epochs={epochs}, batch={batch}, GPUs={num_gpus}")
        if reuse_prepared:
            self._progress(f"Reusing {dataset}; ignoring inputs, partitions and --max-items")
        else:
            self._progress("Preparing masks and images (missing exports are rebuilt)")
            dataset = self.prepare_training_data(inputs, dataset, validation, exclude, max_items=max_items)
        annotations = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
        if not annotations["images"] or not annotations["categories"]:
            raise ValueError("No labelled training images were found in the prepared dataset")
        has_validation = bool(json.loads(
            (dataset / "valid" / "_annotations.coco.json").read_text()
        )["images"])
        device = yolo_device(self.device, num_gpus)
        config = {
            "model": self._model_path(), "threshold": self.threshold,
            "dataset": str(dataset), "has_validation": has_validation,
            "wandb": wandb, "project": f"{self.pipeline.path.name}-{self.name}",
            "run_name": output.name,
            "params": {
                "data": str(output / "yolo-training.json"),
                "project": str(output.parent), "name": output.name, "exist_ok": True,
                "epochs": epochs, "batch": batch * num_gpus, "device": device,
                "imgsz": self.resolution or 640, "overlap_mask": False, "mask_ratio": 1,
                "close_mosaic": 0, "multi_scale": False, "plots": False,
                "save": True, "save_period": -1, "resume": False,
            },
        }
        config_path = output / "yolo-training.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._progress("Starting YOLO trainer (initial weights may download); trainer output follows")
        subprocess.run(
            [sys.executable, "-u", "-c",
             "from collectra.tasks.image_segmentation.yolo_training import launch; import sys; launch(sys.argv[1])",
             str(config_path)],
            check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        checkpoint = output / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO training produced no checkpoint at {checkpoint}")
        destination = self.pipeline.path / f"{self.name}-{uuid.uuid4().hex[:8]}.pt"
        shutil.copy2(checkpoint, destination)
        self.model, self._predictor = destination, None
        self.pipeline.data[self.name]["model"] = destination.name
        self.pipeline.save(self.name)
        self._progress(f"Saved trained checkpoint: {destination}")
        if not keep_log:
            shutil.rmtree(output)
        return destination
