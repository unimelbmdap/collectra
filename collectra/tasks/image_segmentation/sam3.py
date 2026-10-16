"""SAM 3/3.1 image instance segmentation and mask-supervised training data.

Uses Meta's sam3 package, installed separately from its official repository.
Dataset preparation requires only Collectra's ordinary image dependencies.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image as PILImage

from collectra.cli import command
from collectra.tasks.base import Task
from collectra.types.images import Image, ImageSegmentation
from collectra.utils import threading_locked


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


def export_dataset(
    samples, directory: Path, *, prompts=None, validation="", exclude=""
):
    directory = Path(directory).resolve()
    selected = []
    for sample in samples:
        partitions = {item.partition for item in sample.instances if item.partition}
        if sample.image.partition:
            partitions.add(sample.image.partition)
        if len(partitions) > 1:
            raise ValueError(
                "Instances of one image cannot belong to different partitions"
            )
        partition = next(iter(partitions), "")
        if exclude and partition == exclude:
            continue
        selected.append(
            (sample, "valid" if validation and partition == validation else "train")
        )
    labels = sorted({item.name for sample, _ in selected for item in sample.instances})
    if not labels or not any(split == "train" for _, split in selected):
        raise ValueError("No labelled training instances were found")
    categories = [
        {"id": index + 1, "name": (prompts or {}).get(label, label.replace("_", " "))}
        for index, label in enumerate(labels)
    ]
    label_ids = {label: index + 1 for index, label in enumerate(labels)}
    datasets = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "valid")
    }
    for split in datasets:
        (directory / split).mkdir(parents=True, exist_ok=True)
    for image_id, (sample, split) in enumerate(selected, 1):
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
        for instance in sample.instances:
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
    for split, data in datasets.items():
        (directory / split / "_annotations.coco.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
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
        # Official full checkpoints prefix image weights with detector.; fine-
        # tuned image-only checkpoints have raw image-model keys instead.
        full_checkpoint = any(key.startswith("detector.") for key in state)
        if full_checkpoint:
            del state  # the upstream builder reads it; avoid keeping two large copies
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint) if full_checkpoint else None,
            load_from_HF=False,
            device=self.device,
            enable_segmentation=True,
        )
        if not full_checkpoint:
            model.load_state_dict(state, strict=True)
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

    def _samples(self, inputs):
        from collectra.tasks.machine_learning.training import (
            _training_files,
            _prepare_data,
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Attach the task to a pipeline before preparing training data"
            )
        self.pipeline._ensure_workflow_connected()
        node = self.pipeline.node_manager.resolve_node(self.name)
        files = []
        for value in inputs:
            path = Path(value).expanduser().resolve()
            if path.name == "results.yaml" and path.is_file():
                path = path.parent
            if path.is_dir() and (path / "results.yaml").is_file():
                files.append(path)
            else:
                files.extend(_training_files(self, [str(path)]))
        samples = []
        for path in dict.fromkeys(files):
            # Read each results directory independently. Different datasets can
            # have the same page basename and even the same artefact ids.
            targets, target_maps = _prepare_data(
                self, [path], self.pipeline.node_manager.get_children_artefact(node)
            )
            parents, parent_maps = _prepare_data(
                self, [path], self.pipeline.node_manager.get_ancestor_artefacts(node)
            )
            samples.extend(
                prepare_segmentation_inputs(targets, parents, target_maps, parent_maps)
            )
        return samples

    @command
    def prepare_training_data(
        self, inputs: list[str], output: Path, validation: str = "", exclude: str = ""
    ) -> Path:
        """Export annotated masks and input images as SAM-compatible COCO datasets."""
        return export_dataset(
            self._samples(inputs),
            output,
            prompts=self._prompts(),
            validation=validation,
            exclude=exclude,
        )

    @command
    def train(
        self,
        inputs: list[str],
        output: Path,
        validation: str = "",
        exclude: str = "",
        epochs: int = 20,
        batch: int = 1,
        num_gpus: int = 1,
    ) -> Path:
        """Prepare instance masks and fine-tune using Meta's image trainer."""
        if min(epochs, batch, num_gpus) < 1:
            raise ValueError("epochs, batch and num_gpus must be positive")
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        dataset = self.prepare_training_data(
            inputs, output / "dataset", validation, exclude
        )
        from .sam3_training import training_config

        config = training_config(
            dataset,
            output,
            self._checkpoint(),
            epochs=epochs,
            batch=batch,
            num_gpus=num_gpus,
        )
        config_path = output / "sam3-training.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from collectra.tasks.image_segmentation.sam3_training import launch; import sys; launch(sys.argv[1])",
                str(config_path),
            ],
            check=True,
        )
        checkpoint = output / "training" / "checkpoints" / "checkpoint.pt"
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
        return destination
