"""SAM 3/3.1 image instance segmentation and mask-supervised training data.

Uses Meta's sam3 package, installed separately from its official repository.
Dataset preparation requires only Collectra's ordinary image dependencies.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from cappa import Arg
import numpy as np
import yaml
from PIL import Image as PILImage

from collectra.cli import command
from collectra.tasks.base import Task
from collectra.types.images import Image, ImageSegmentation
from collectra.utils import threading_locked


def training_progress(message):
    """Keep progress visible in terminals and redirected cluster job logs."""
    print(f"[SAM3] {message}", flush=True)


def _sam3():
    try:
        import sam3
    except ModuleNotFoundError as error:
        if error.name != "sam3":
            raise
        raise ImportError(
            'Install Meta SAM3 separately: pip install "sam3[train] @ '
            'git+https://github.com/facebookresearch/sam3.git". '
            "See Meta prerequisites (Python 3.12+, PyTorch, CUDA) and checkpoint access."
        ) from error
    return sam3


# Keep these imports available for callers using the original SAM3 module.
from .data import (
    SegmentationSample, SegmentationTrainingData, _bounds, _sample_partition,
    encode_rle, export_dataset, mask_in_view, mask_to_source,
    prepare_segmentation_inputs,
)


class ImageSegmentationSam3(SegmentationTrainingData, Task):
    """Find all instances matching text prompts and return one mask per instance."""

    def __init__(
        self,
        name: str,
        model: str | Path = "",
        prompt: str = "",
        prompts: dict[str, str] | None = None,
        version: str = "sam3.1",
        threshold: float = 0.5,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        if version not in {"sam3", "sam3.1"}:
            raise ValueError("version must be sam3 or sam3.1")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.model, self.prompt, self.prompts = model, prompt, prompts or {}
        self.version, self.threshold, self.device = version, threshold, device
        self._processor = None

    def attributes_to_ignore(self):
        return super().attributes_to_ignore() | {"_processor"}

    def input_type(self) -> type:
        return Image

    def output_type(self) -> type:
        return ImageSegmentation

    _progress = staticmethod(training_progress)

    def _prompts(self):
        if self.prompts:
            mapping = self.prompts
        else:
            outputs = getattr(self, "output", None) or [self.get_output_name()]
            outputs = outputs if isinstance(outputs, list) else [outputs]
            mapping = {
                label: self.prompt or label.replace("_", " ") for label in outputs
            }
        if any(not label or not prompt.strip() for label, prompt in mapping.items()):
            raise ValueError("SAM prompts and output labels must be nonempty")
        return mapping

    def _checkpoint(self):
        _sam3()
        if self.model:
            path = Path(self.model).expanduser()
            if not path.is_absolute() and self.pipeline is not None:
                path = Path(self.pipeline.path) / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        from sam3.model_builder import download_ckpt_from_hf

        return Path(download_ckpt_from_hf(version=self.version))

    def _init_model(self):
        if self._processor is not None:
            return
        _sam3()
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = self._checkpoint()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        from .sam3_training import load_image_state

        model = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=False,
            device=self.device,
            enable_segmentation=True,
        )
        load_image_state(model, state)
        model.to(self.device).eval()
        self._processor = Sam3Processor(
            model, device=self.device, confidence_threshold=self.threshold
        )

    @threading_locked()
    def run(self, *images: Image) -> list[ImageSegmentation]:
        if not images or any(not isinstance(image, Image) for image in images):
            raise TypeError(
                "SAM requires one or more Image artefacts (including subclasses)"
            )
        prompts = self._prompts()
        self._init_model()
        directory = Path(self.context.file_path or Path.cwd()) / "masks"
        directory.mkdir(parents=True, exist_ok=True)
        import torch

        results = []
        # SAM3's fused ViT MLP kernel (addmm_act) always returns bfloat16, so
        # the following float32 layers need bfloat16 autocast, as in Meta's
        # own inference examples; it also requires gradients to be disabled.
        device_type = torch.device(self.device).type
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16):
            for image in images:
                state = self._processor.set_image(image.pil().convert("RGB"))
                for label, prompt in prompts.items():
                    self._processor.reset_all_prompts(state)
                    prediction = self._processor.set_text_prompt(prompt=prompt, state=state)
                    masks = prediction["masks"]
                    if hasattr(masks, "detach"):
                        # Under autocast masks may be bfloat16, which NumPy lacks.
                        masks = masks.detach().float().cpu().numpy()
                    masks = np.asarray(masks)
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    if masks.ndim != 3:
                        raise ValueError("SAM returned masks with an unexpected shape")
                    for mask in masks:
                        pixels = mask_to_source(image, mask > 0.5)
                        if not pixels.any():
                            continue
                        path = directory / f"sam3-{uuid.uuid4().hex}.png"
                        PILImage.fromarray(pixels).save(path)
                        results.append(
                            ImageSegmentation(
                                name=label,
                                data=image.get_path(),
                                mask=path,
                                parents=[image.id],
                                orientation=image.orientation,
                                partition=image.partition,
                            )
                        )
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
        """Prepare instance masks and fine-tune using Meta's image trainer.

        --reuse-data uses OUTPUT/dataset directly when present, otherwise
        prepares it from inputs. It requires --output.
        """
        if min(epochs, batch, num_gpus) < 1:
            raise ValueError("epochs, batch and num_gpus must be positive")
        if reuse_data and output is None:
            raise ValueError("--reuse-data requires --output")
        output = (
            Path(output).expanduser().resolve()
            if output is not None
            else Path.cwd() / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        reuse_prepared = reuse_data and all(
            (output / "dataset" / split / "_annotations.coco.json").is_file()
            for split in ("train", "valid")
        )
        if not reuse_prepared and max_items is not None and max_items < 1:
            raise ValueError("max_items must be positive")
        if not reuse_prepared and not inputs:
            raise ValueError(
                "Supply training inputs to prepare the dataset; --reuse-data without "
                f"inputs requires an existing export at {output / 'dataset'}"
            )
        output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        training_progress(
            f"Training {self.name}: output={output}, epochs={epochs}, batch={batch}, GPUs={num_gpus}"
        )
        if reuse_prepared:
            dataset = output / "dataset"
            training_progress(f"Reusing prepared dataset directly: {dataset}")
            training_progress("Skipping source checks and export; using existing train/validation splits")
            if max_items is not None:
                training_progress("Ignoring --max-items because --reuse-data uses the existing dataset unchanged")
        else:
            if reuse_data:
                training_progress("Prepared dataset is missing or incomplete; falling back to preparation from inputs")
            training_progress("Preparing the dataset before GPU training starts")
            dataset = self.prepare_training_data(
                inputs, output / "dataset", validation, exclude, max_items=max_items
            )
        from .sam3_training import training_config

        training_progress("Resolving the SAM checkpoint (a download may be needed)")
        initial_checkpoint = self._checkpoint()
        training_progress(f"Using checkpoint: {initial_checkpoint}")
        training_progress("Building the SAM training configuration")
        config = training_config(
            dataset,
            output,
            initial_checkpoint,
            epochs=epochs,
            batch=batch,
            num_gpus=num_gpus,
            wandb=wandb,
            run_name=output.name,
            project=f"{self.pipeline.name}-{self.name}",
            confidence_threshold=self.threshold,
        )
        config_path = output / "sam3-training.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        training_progress(f"Configuration saved: {config_path}")
        training_progress(
            f"Starting SAM trainer ({time.monotonic() - started:.1f}s preparation); "
            "trainer output follows"
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                "from collectra.tasks.image_segmentation.sam3_training import launch; import sys; launch(sys.argv[1])",
                str(config_path),
            ],
            check=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        training_progress("SAM trainer exited successfully; checking its checkpoint")
        checkpoint = output / "training" / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM training produced no checkpoint at {checkpoint}"
            )
        destination = self.pipeline.path / f"{self.name}-{uuid.uuid4().hex[:8]}.pt"
        shutil.copy2(checkpoint, destination)
        self.model = destination
        self._processor = None
        self.pipeline.data[self.name]["model"] = destination.name
        self.pipeline.save(self.name)
        training_progress(f"Saved trained checkpoint: {destination}")
        if not keep_log:
            shutil.rmtree(output, ignore_errors=True)
            training_progress(f"Removed training directory: {output}")
        training_progress(f"Training complete ({time.monotonic() - started:.1f}s)")
        return destination
