"""Object detection with SAM 3, from text prompts.

Uses Meta's sam3 package, installed separately from its official repository.
Each output of the task is a kind of object: it is found by prompting SAM 3 with
the output's name (or the prompt given for it), and every box found is returned
as a crop with that name.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from cappa import Arg
from pydantic import Field, PrivateAttr

from collectra.cli import command
from collectra.types.images import Image, ImageCrop
from collectra.utils import change_dir, threading_locked

from ..image_segmentation.sam3 import Sam3Backend, training_progress
from ..machine_learning.training import (
    prepare_object_detection_inputs,
    run_training_command,
)
from .base import Batch, Epochs, ObjectDetector
from .coco_data import CocoDetectionData
from .detr import DetectionTrainResult

__all__ = ["ObjectDetectorSam3"]


class ObjectDetectorSam3(Sam3Backend, CocoDetectionData, ObjectDetector):
    """Find objects by describing them in words.

    Attributes:
        model: A fine-tuned SAM 3 checkpoint. Empty uses Meta's released weights.
        prompt: The text prompt for every output. Empty uses each output's own
            name, with underscores as spaces.
        prompts: Output name to text prompt, for outputs that need their own.
        version: Which of Meta's released weights to download: sam3 or sam3.1.
        threshold: Minimum confidence for a detection, from 0 to 1.
        device: Where to run the model. SAM 3 needs a CUDA GPU.

    SAM 3 always works at its own image size, so ``resolution`` has no effect.
    """

    model: str | Path = ""
    prompt: str = ""
    prompts: dict[str, str] = Field(default_factory=dict)
    version: Literal["sam3", "sam3.1"] = "sam3.1"
    threshold: float = Field(default=0.5, ge=0, le=1)
    device: str = "cuda"

    _processor: Any = PrivateAttr(default=None)

    def attributes_to_ignore(self):
        return super().attributes_to_ignore() | {"_processor"}

    def input_type(self) -> type:
        return Image

    def output_type(self) -> type:
        return ImageCrop

    @threading_locked()
    def run(self, *args: Image) -> list[ImageCrop]:
        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        if not isinstance(args[0], Image):
            raise TypeError("Input must be an instance of Image.")
        image = args[0]
        prompts = self._prompts()
        self._init_model()
        self._processor.set_confidence_threshold(self.threshold)

        import torch

        picture = image.pil().convert("RGB")
        width, height = picture.size
        crops: list[ImageCrop] = []
        # SAM 3's fused kernels need bfloat16 autocast and no gradients.
        device_type = torch.device(self.device).type
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16):
            state = self._processor.set_image(picture)
            for label, prompt in prompts.items():
                self._processor.reset_all_prompts(state)
                found = self._processor.set_text_prompt(prompt=prompt, state=state)
                boxes = found["boxes"].detach().float().cpu().numpy()
                scores = found["scores"].detach().float().cpu().numpy()
                for (x1, y1, x2, y2), score in zip(boxes, scores):
                    x1, x2 = (max(0.0, min(float(width), float(x))) for x in (x1, x2))
                    y1, y2 = (max(0.0, min(float(height), float(y))) for y in (y1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crops.append(
                        image.make_crop(
                            x_center=(x1 + x2) / 2.0 / width,
                            y_center=(y1 + y2) / 2.0 / height,
                            width_relative=(x2 - x1) / width,
                            height_relative=(y2 - y1) / height,
                            confidence=float(score),
                            orientation=image.orientation,
                            name=label,
                        )
                    )
        return self._orient_crops(crops)

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                help="Pipeline data files or directories containing training annotations."
            ),
        ],
        output: Annotated[
            Path | None,
            Arg(help="Directory for training logs, dataset files, and checkpoints."),
        ] = None,
        keep_log: Annotated[
            bool,
            Arg(
                help="Keep the training directory after saving the model to the pipeline."
            ),
        ] = True,
        validation: Annotated[
            str,
            Arg(
                help="Partition value identifying validation examples. The best model is the one with the highest box average precision on them."
            ),
        ] = "",
        exclude: Annotated[
            str,
            Arg(help="Partition value identifying examples to exclude from training."),
        ] = "",
        epochs: Epochs = None,
        batch: Batch = None,
        num_gpus: Annotated[
            int, Arg(help="Number of GPUs used for training on this machine.")
        ] = 1,
        wandb: Annotated[
            bool,
            Arg(help="Log training metrics to Weights & Biases."),
        ] = False,
        model: Annotated[
            str,
            Arg(
                help="SAM 3 checkpoint to start from. Omit to keep the configured model, or else Meta's released weights."
            ),
        ] = "",
        max_items: Annotated[
            int,
            Arg(
                help="Use at most this many images in each of the training and validation sets (0 uses all); for quick tests."
            ),
        ] = 0,
    ):
        """Fine-tune SAM 3 on annotated boxes, with Meta's trainer."""
        settings = self._settings_for_run(epochs=epochs, batch=batch)
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
            epochs=settings["epochs"],
            batch=settings["batch"],
            num_gpus=num_gpus,
            wandb=wandb,
            model=model,
            max_items=max_items,
        )

    def _train(self, *images: ImageCrop, **kwargs) -> DetectionTrainResult:
        if kwargs.get("model"):
            self.model = kwargs["model"]
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

        classes = list(kwargs.get("classes") or self._declared_outputs())
        classes = classes or sorted(
            {img.name for img in images if isinstance(img, ImageCrop)}
        )
        if not classes:
            raise ValueError("No classes provided/found for SAM 3 training.")
        prompts = self._prompts_for(classes)
        if len(set(prompts.values())) != len(classes):
            raise ValueError(
                "SAM 3 learns each class from its prompt, so every output needs a "
                f"different prompt; got {prompts}"
            )

        train_samples, val_samples = self._prepare_assets(
            classes,
            kwargs.get("validation", ""),
            kwargs.get("exclude", ""),
            *images,
        )
        max_items = int(kwargs.get("max_items", 0) or 0)
        if max_items > 0:
            train_samples = train_samples[:max_items]
            val_samples = val_samples[:max_items]
        self._check_distribution(classes, train_samples, val_samples)
        with change_dir(kwargs["base_folder"]):
            dataset = self._write_coco_dataset(
                train_samples, val_samples, classes, log / "dataset"
            )
            name_categories(dataset, prompts)
            checkpoint = self._run_trainer(
                dataset,
                log,
                masks=False,
                epochs=int(kwargs.get("epochs", self.epochs)),
                batch=int(kwargs.get("batch", self.batch)),
                num_gpus=int(kwargs.get("num_gpus", 1)),
                wandb=bool(kwargs.get("wandb", False)),
                project=kwargs.get("project"),
                started=time.monotonic(),
            )
        weights = log / "weights"
        weights.mkdir(exist_ok=True)
        shutil.copy2(checkpoint, weights / "best.pt")
        training_progress(f"Best SAM 3 model saved to: {weights / 'best.pt'}")
        return DetectionTrainResult(
            save_dir=log, results_dict={"status": "completed", "classes": classes}
        )


def name_categories(dataset: Path, prompts: dict[str, str]) -> None:
    """Rename each COCO category to its prompt: SAM 3 learns a class by its words."""
    for split in ("train", "valid"):
        path = dataset / split / "_annotations.coco.json"
        data = json.loads(path.read_text())
        for category in data["categories"]:
            category["name"] = prompts[category["name"]]
        path.write_text(json.dumps(data, indent=2))
