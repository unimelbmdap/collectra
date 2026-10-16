"""SAM 3/3.1 image instance segmentation and mask-supervised training data.

Uses Meta's sam3 package, installed separately from its official repository.
Dataset preparation requires only Collectra's ordinary image dependencies.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
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


def _bounds(image: Image):
    return tuple(round(value) for value in image.display_bounds())


def mask_in_view(image: Image, mask: PILImage.Image):
    """Apply the same crop, region and rotation as image.pil()."""
    if mask.size != (image.raw_width, image.raw_height):
        raise ValueError("Annotation mask dimensions differ from the source image")
    if isinstance(image, ImageSegmentation):
        pixels = (np.asarray(mask) != 0) & (np.asarray(image._mask_image()) != 0)
        mask = PILImage.fromarray(pixels.astype("uint8"))
    return mask.crop(_bounds(image)).rotate(image.orientation.to_degree(), expand=True)


def mask_to_source(image: Image, mask: np.ndarray):
    """Invert the displayed image's rotation/crop into source-sized 0/1 pixels."""
    local = PILImage.fromarray(mask.astype("uint8"))
    local = local.rotate(-image.orientation.to_degree(), expand=True)
    left, top, right, bottom = _bounds(image)
    if local.size != (right - left, bottom - top):
        raise ValueError("Predicted mask dimensions differ from the input image view")
    source = PILImage.new("L", (image.raw_width, image.raw_height))
    source.paste(local, (left, top))
    pixels = np.asarray(source) != 0
    if isinstance(image, ImageSegmentation):
        pixels &= np.asarray(image._mask_image()) != 0
    return pixels.astype("uint8")


def encode_rle(mask: np.ndarray):
    """Lossless COCO uncompressed RLE, including holes and disconnected pixels."""
    values = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    boundaries = np.concatenate(
        ([0], np.flatnonzero(values[1:] != values[:-1]) + 1, [len(values)])
    )
    counts = np.diff(boundaries).tolist()
    if len(values) and values[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


@dataclass
class SegmentationSample:
    image: Image
    instances: list[ImageSegmentation]


def prepare_segmentation_inputs(
    processed_inputs, processed_parents, input_maps, parent_input_maps
):
    samples = []
    for filename, instances in input_maps.items():
        parents = {
            image.id: image
            for image in parent_input_maps.get(filename, [])
            if isinstance(image, Image)
        }
        grouped = {}
        for instance in instances:
            if not isinstance(instance, ImageSegmentation):
                raise TypeError(
                    "SAM training targets must be ImageSegmentation instances"
                )
            ids = (
                [instance.parents]
                if isinstance(instance.parents, str)
                else instance.parents
            )
            candidates = [parents[key] for key in ids if key in parents]
            if len(candidates) != 1:
                raise ValueError(
                    f"Mask {instance.id!r} must identify exactly one input image parent"
                )
            parent = candidates[0]
            if parent.get_path().resolve() != instance.get_path().resolve():
                raise ValueError(
                    f"Mask {instance.id!r} refers to a different source image"
                )
            grouped.setdefault(
                parent.id, SegmentationSample(parent, [])
            ).instances.append(instance)
        samples.extend(grouped.values())
    return samples


def _sample_partition(sample):
    partitions = {item.partition for item in sample.instances if item.partition}
    if sample.image.partition:
        partitions.add(sample.image.partition)
    if len(partitions) > 1:
        raise ValueError("Instances of one image cannot belong to different partitions")
    return next(iter(partitions), "")


def export_dataset(
    samples, directory: Path, *, prompts=None, validation="", exclude="", max_items=None
):
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    started = time.monotonic()
    directory = Path(directory).resolve()
    selected = []
    for sample in samples:
        partition = _sample_partition(sample)
        if exclude and partition == exclude:
            continue
        selected.append(
            (sample, "valid" if validation and partition == validation else "train")
        )
        if max_items is not None and len(selected) >= max_items:
            break
    labels = sorted({item.name for sample, _ in selected for item in sample.instances})
    if not labels or not any(split == "train" for _, split in selected):
        raise ValueError("No labelled training instances were found")
    categories = [
        {"id": index + 1, "name": (prompts or {}).get(label, label.replace("_", " "))}
        for index, label in enumerate(labels)
    ]
    label_ids = {label: index + 1 for index, label in enumerate(labels)}
    training_progress(
        f"Exporting {len(selected)} images and "
        f"{sum(len(sample.instances) for sample, _ in selected)} masks to {directory} "
        f"({sum(split == 'train' for _, split in selected)} train, "
        f"{sum(split == 'valid' for _, split in selected)} validation images)"
    )
    datasets = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "valid")
    }
    for split in datasets:
        (directory / split).mkdir(parents=True, exist_ok=True)
    for image_id, (sample, split) in enumerate(selected, 1):
        training_progress(
            f"Export {image_id}/{len(selected)} [{split}]: {sample.image.id} "
            f"({len(sample.instances)} masks; {time.monotonic() - started:.1f}s elapsed)"
        )
        image = sample.image.pil().convert("RGB")
        filename = f"{image_id:08d}.png"
        image.save(directory / split / filename)
        data = datasets[split]
        data["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": image.width,
                "height": image.height,
            }
        )
        for mask_index, instance in enumerate(sample.instances, 1):
            mask = np.asarray(mask_in_view(sample.image, instance._mask_image())) != 0
            ys, xs = np.nonzero(mask)
            if not len(xs):
                raise ValueError(
                    f"Mask {instance.id!r} has no pixels inside its input image"
                )
            if mask.shape != (image.height, image.width):
                raise ValueError(
                    "Image subclass crop differs from its declared display bounds"
                )
            data["annotations"].append(
                {
                    "id": len(data["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": label_ids[instance.name],
                    "iscrowd": 0,
                    "bbox": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ],
                    "area": int(mask.sum()),
                    "segmentation": encode_rle(mask),
                }
            )
            if mask_index % 25 == 0:
                training_progress(
                    f"  Image {image_id}/{len(selected)}: encoded mask "
                    f"{mask_index}/{len(sample.instances)}"
                )
    for split, data in datasets.items():
        training_progress(
            f"Writing {split} annotations: {len(data['images'])} images, "
            f"{len(data['annotations'])} masks"
        )
        (directory / split / "_annotations.coco.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
    training_progress(f"Dataset export complete ({time.monotonic() - started:.1f}s): {directory}")
    return directory


class ImageSegmentationSam3(Task):
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
        results = []
        for image in images:
            state = self._processor.set_image(image.pil().convert("RGB"))
            for label, prompt in prompts.items():
                self._processor.reset_all_prompts(state)
                prediction = self._processor.set_text_prompt(prompt=prompt, state=state)
                masks = prediction["masks"]
                if hasattr(masks, "detach"):
                    masks = masks.detach().cpu().numpy()
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

    def _samples(self, inputs, max_items=None, exclude=""):
        from collectra.tasks.machine_learning.training import (
            _training_files,
            _prepare_data,
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Attach the task to a pipeline before preparing training data"
            )
        started = time.monotonic()
        training_progress("Discovering training result files (CPU/I/O preparation)")
        self.pipeline._ensure_workflow_connected()
        node = self.pipeline.node_manager.resolve_node(self.name)
        files = []
        for value in inputs:
            path = Path(value).expanduser().resolve()
            training_progress(f"Scanning {path}")
            if path.name == "results.yaml" and path.is_file():
                path = path.parent
            if path.is_dir() and (path / "results.yaml").is_file():
                files.append(path)
            else:
                files.extend(_training_files(self, [str(path)]))
        samples = []
        files = list(dict.fromkeys(files))
        training_progress(f"Found {len(files)} result files; loading images and validating masks")
        for index, path in enumerate(files, 1):
            training_progress(
                f"Load {index}/{len(files)}: {path} "
                f"({time.monotonic() - started:.1f}s elapsed)"
            )
            # Read each results directory independently. Different datasets can
            # have the same page basename and even the same artefact ids.
            targets, target_maps = _prepare_data(
                self, [path], self.pipeline.node_manager.get_children_artefact(node)
            )
            parents, parent_maps = _prepare_data(
                self, [path], self.pipeline.node_manager.get_ancestor_artefacts(node)
            )
            for sample in prepare_segmentation_inputs(targets, parents, target_maps, parent_maps):
                if max_items is not None and exclude and _sample_partition(sample) == exclude:
                    continue
                samples.append(sample)
                if max_items is not None and len(samples) >= max_items:
                    break
            if max_items is not None and len(samples) >= max_items:
                training_progress(f"Reached max_items={max_items}; stopping source loading")
                break
        training_progress(
            f"Loaded {len(samples)} labelled images and "
            f"{sum(len(sample.instances) for sample in samples)} masks "
            f"({time.monotonic() - started:.1f}s)"
        )
        return samples

    def prepare_training_data(
        self, inputs: list[str], output: Path, validation: str = "", exclude: str = "",
        max_items: int | None = None,
    ) -> Path:
        """Export annotated masks and input images as SAM-compatible COCO datasets."""
        return export_dataset(
            self._samples(inputs, max_items=max_items, exclude=exclude),
            output,
            prompts=self._prompts(),
            validation=validation,
            exclude=exclude,
            max_items=max_items,
        )

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
            project=f"{self.pipeline.path.name}-{self.name}",
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
