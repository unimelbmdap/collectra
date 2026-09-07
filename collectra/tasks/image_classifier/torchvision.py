"""Torchvision image classification with portable, metadata-bearing checkpoints."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cappa import Arg

from collectra.cli import command
from collectra.logger import get_logger
from collectra.types.images import Image, ImageCrop
from collectra.types.links import Link
from collectra.utils import threading_locked
from ..base import Task
from ..machine_learning.training import print_distribution_table, run_training_command
from .training import prepare_classification_inputs

__all__ = ["ImageClassifierTorchvision"]
logger = get_logger(__name__)


@dataclass
class ClassificationTrainResult:
    save_dir: Path
    results_dict: dict


def _replace_head(model, num_classes: int):
    """Replace the output projection for torchvision's classification families."""
    from torch import nn

    for name in ("fc", "classifier", "heads", "head"):
        head = getattr(model, name, None)
        if head is None:
            continue
        if isinstance(head, nn.Linear):
            replacement = nn.Linear(
                head.in_features, num_classes, bias=head.bias is not None
            )
            setattr(model, name, replacement)
            return replacement
        if isinstance(head, nn.Sequential):
            projections = [
                (path, module)
                for path, module in head.named_modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))
            ]
            if not projections:
                continue
            path, layer = projections[-1]
            if isinstance(layer, nn.Linear):
                replacement = nn.Linear(
                    layer.in_features, num_classes, bias=layer.bias is not None
                )
            else:  # SqueezeNet's classifier is a 1x1 convolution.
                replacement = nn.Conv2d(
                    layer.in_channels,
                    num_classes,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=layer.bias is not None,
                )
                model.num_classes = num_classes
            parent_path, _, child = path.rpartition(".")
            parent = head.get_submodule(parent_path) if parent_path else head
            setattr(parent, child, replacement)
            return replacement
    raise ValueError(f"Unsupported classification head in {type(model).__name__}")


def _disable_auxiliary_heads(model) -> None:
    # GoogLeNet and Inception should return a single logits tensor in training too.
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    for name in ("AuxLogits", "aux1", "aux2"):
        if hasattr(model, name):
            setattr(model, name, None)


class ImageClassifierTorchvision(Task):
    """Train torchvision classifiers and return predicted class Links."""

    prepare_training_inputs = staticmethod(prepare_classification_inputs)

    def __init__(self, name: str, model: str | Path = "resnet18", **kwargs):
        super().__init__(name, model=model, **kwargs)
        self.original_model_path: Path | None = None
        self._categories: list[str] = []
        self._architecture = ""
        self._model_kwargs: dict = {}
        self._preprocessing: dict = {}
        self._device = "cpu"

    @staticmethod
    def _select_device(device: str = "") -> str:
        import torch

        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _transforms(self, *, training: bool = False, fliplr: float = 0.0):
        from torchvision import transforms as T

        config = self._preprocessing
        interpolation = T.InterpolationMode(config["interpolation"])
        steps = [
            T.Resize(
                config["resize_size"], interpolation=interpolation, antialias=True
            ),
            T.CenterCrop(config["crop_size"]),
        ]
        if training and fliplr:
            steps.append(T.RandomHorizontalFlip(fliplr))
        steps.extend([T.ToTensor(), T.Normalize(config["mean"], config["std"])])
        return T.Compose(steps)

    def _load(self, *, pretrained: bool = True) -> None:
        import torch
        from torchvision import models

        if isinstance(self.model, torch.nn.Module):
            return
        if not isinstance(self.model, (str, Path)):
            raise ValueError(
                "Model must be a torchvision model name or a checkpoint path"
            )
        path = Path(self.model).expanduser()
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            required = {
                "architecture",
                "state_dict",
                "classes",
                "preprocessing",
                "model_kwargs",
            }
            if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
                raise ValueError(
                    "Expected a Collectra torchvision checkpoint containing architecture, "
                    "state_dict, classes, preprocessing, and model_kwargs. "
                    "A bare state_dict does not identify its architecture or class labels."
                )
            self._architecture = checkpoint["architecture"]
            self._model_kwargs = checkpoint["model_kwargs"]
            self._categories = checkpoint["classes"]
            self._preprocessing = checkpoint["preprocessing"]
            model = models.get_model(
                self._architecture, weights=None, **self._model_kwargs
            )
            _disable_auxiliary_heads(model)
            _replace_head(model, len(self._categories))
            model.load_state_dict(checkpoint["state_dict"])
            self.original_model_path = path.resolve()
        else:
            name = str(self.model)
            available = models.list_models(module=models)
            if name not in available:
                raise ValueError(
                    f"Unknown torchvision classification model or missing checkpoint: {name!r}. "
                    f"Available models: {', '.join(available)}"
                )
            weights = models.get_model_weights(name).DEFAULT
            preset = weights.transforms()
            self._preprocessing = {
                "crop_size": preset.crop_size,
                "resize_size": preset.resize_size,
                "mean": preset.mean,
                "std": preset.std,
                "interpolation": preset.interpolation.value,
            }
            builder_kwargs = {}
            if name.startswith("vit_"):
                builder_kwargs["image_size"] = preset.crop_size[0]
            if name in {"inception_v3", "googlenet"}:
                builder_kwargs["init_weights"] = False
            model = models.get_model(
                name, weights=weights if pretrained else None, **builder_kwargs
            )
            _disable_auxiliary_heads(model)
            self._architecture = name
            self._model_kwargs = dict(builder_kwargs)
            if hasattr(model, "transform_input"):
                self._model_kwargs["transform_input"] = model.transform_input
            if hasattr(model, "aux_logits"):
                self._model_kwargs["aux_logits"] = False
            if hasattr(model, "image_size"):
                self._model_kwargs["image_size"] = model.image_size
            self._categories = list(weights.meta["categories"])
            self.original_model_path = None
        self.model = model.to(self._device)
        self.model.eval()

    def _reload(self) -> None:
        if self.original_model_path is None:
            raise ValueError("No checkpoint is available to reload")
        self.model = self.original_model_path
        self._load()

    @threading_locked()
    def run(self, *args: Image) -> Link:
        """Classify one image or crop and link the predicted class to it."""
        import torch

        if len(args) != 1 or not isinstance(args[0], Image):
            raise ValueError("This task requires a single Image or ImageCrop input")
        self._device = self._select_device(getattr(self, "device", ""))
        self._load()
        self.model.to(self._device).eval()
        image = args[0]
        with image.pil() as pixels:
            batch = (
                self._transforms()(pixels.convert("RGB")).unsqueeze(0).to(self._device)
            )
        with torch.inference_mode():
            class_index = self.model(batch).argmax(dim=1).item()
        class_name = self._categories[class_index]
        outputs = getattr(self, "output", [])
        outputs = [outputs] if isinstance(outputs, str) else outputs
        if outputs and class_name not in outputs:
            raise ValueError(
                f"Predicted class {class_name!r} is not a configured output: {outputs}"
            )
        return Link(name=class_name, target=image)

    def _prepare_assets(self, classes, log, validation, exclude, *images, min_size=0):
        """Materialize oriented RGB images and crops in train/<class> and val/<class>."""
        if min_size < 0:
            raise ValueError("min_size cannot be negative")
        for name in classes:
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError(f"Class name must be a directory name: {name!r}")
        train_dir, val_dir = log / "train", log / "val"
        for directory in (train_dir, val_dir):
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(
                    f"Dataset directory is not empty: {directory}. Use a new output directory."
                )
        counts = {split: {name: 0 for name in classes} for split in ("train", "val")}
        for directory in (train_dir, val_dir):
            for name in classes:
                (directory / name).mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            if exclude and image.partition == exclude:
                continue
            if image.name not in classes:
                raise ValueError(
                    f"Training label {image.name!r} is not a configured class"
                )
            target = image.resolve() if type(image) is Link else image
            if not isinstance(target, Image):
                raise TypeError(
                    f"Classifier input {image.id!r} does not resolve to an Image"
                )
            if isinstance(target, ImageCrop):
                left, top, right, bottom = target.coordinates()
                if right <= left or bottom <= top:
                    logger.warning("Skipping empty classifier crop %s", image.id)
                    continue
            with target.pil() as pixels:
                if min(pixels.size) < max(1, min_size):
                    logger.warning(
                        "Skipping classifier image %s below minimum size %d",
                        image.id,
                        min_size,
                    )
                    continue
                split = (
                    "val" if validation and image.partition == validation else "train"
                )
                destination = log / split / image.name / f"{index:08d}.png"
                pixels.convert("RGB").save(destination)
                counts[split][image.name] += 1
        print_distribution_table(
            "Class Distribution", classes, counts["train"], counts["val"]
        )
        missing = [name for name in classes if not counts["train"][name]]
        if missing:
            raise ValueError(f"No training images for classes: {', '.join(missing)}")
        return train_dir, val_dir

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                help="Pipeline data files or directories containing classification labels."
            ),
        ],
        model: Annotated[
            str,
            Arg(
                help="Torchvision classification model name (e.g. resnet18, efficientnet_b0, vit_b_16) or a Collectra torchvision checkpoint. Omit to keep the configured model."
            ),
        ] = "",
        output: Annotated[
            Path | None,
            Arg(
                help="Directory for extracted train/val images, metrics, and checkpoints."
            ),
        ] = None,
        keep_log: Annotated[
            bool,
            Arg(
                help="Keep extracted images and training logs after saving the model to the pipeline."
            ),
        ] = True,
        validation: Annotated[
            str,
            Arg(
                help="Partition value identifying validation images; omit to train without validation."
            ),
        ] = "",
        exclude: Annotated[
            str, Arg(help="Partition value identifying images to exclude.")
        ] = "",
        epochs: Annotated[int, Arg(help="Number of training epochs.")] = 10,
        batch: Annotated[int, Arg(help="Number of images per training batch.")] = 16,
        learning_rate: Annotated[
            float, Arg(help="Initial learning rate for AdamW.")
        ] = 0.0001,
        weight_decay: Annotated[
            float, Arg(help="Weight decay used for regularization.")
        ] = 0.0001,
        pretrained: Annotated[
            bool,
            Arg(
                help="Use DEFAULT pretrained weights for a named model; checkpoints always load their saved weights."
            ),
        ] = True,
        freeze_backbone: Annotated[
            bool,
            Arg(
                help="Train only the final classification layer; freeze all other parameters."
            ),
        ] = False,
        early_stop: Annotated[
            int,
            Arg(
                help="Stop after this many epochs without validation-loss improvement; 0 disables."
            ),
        ] = 10,
        min_size: Annotated[
            int,
            Arg(
                help="Skip images or crops with either dimension below this pixel threshold; 0 disables."
            ),
        ] = 0,
        fliplr: Annotated[
            float,
            Arg(
                help="Probability of horizontal flipping during training, from 0 to 1."
            ),
        ] = 0.5,
        workers: Annotated[
            int, Arg(help="Number of data-loading worker processes.")
        ] = 0,
        device: Annotated[
            str,
            Arg(
                help="Torch device such as cpu, cuda, cuda:0, or mps; omit for automatic selection."
            ),
        ] = "",
        seed: Annotated[
            int, Arg(help="Random seed for model initialization and data loading.")
        ] = 0,
    ):
        """Fine-tune a torchvision image classifier using its weights' prescribed image preprocessing."""
        result = run_training_command(
            self,
            inputs,
            self._train,
            keep_log=keep_log,
            output=output,
            validation=validation,
            exclude=exclude,
            prepare_inputs=self.prepare_training_inputs,
            model=model,
            epochs=epochs,
            batch=batch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            early_stop=early_stop,
            min_size=min_size,
            fliplr=fliplr,
            workers=workers,
            device=device,
            seed=seed,
        )
        if self.pipeline is not None:
            saved_model = self.pipeline.data[self.name]["model"]
            self.original_model_path = (self.pipeline.path / saved_model).resolve()
        return result

    def _checkpoint(self, epoch: int, metrics: dict) -> dict:
        return {
            "format_version": 1,
            "architecture": self._architecture,
            "model_kwargs": self._model_kwargs,
            "classes": self._categories,
            "preprocessing": self._preprocessing,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "epoch": epoch,
            "metrics": metrics,
        }

    def _train(self, *images: Image, **kwargs) -> ClassificationTrainResult:
        import torch
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        epochs, batch = int(kwargs.get("epochs", 10)), int(kwargs.get("batch", 16))
        workers, patience = int(kwargs.get("workers", 0)), int(
            kwargs.get("early_stop", 10)
        )
        fliplr = float(kwargs.get("fliplr", 0.5))
        if (
            epochs < 1
            or batch < 1
            or workers < 0
            or patience < 0
            or not 0 <= fliplr <= 1
        ):
            raise ValueError(
                "epochs and batch must be positive; workers/early_stop nonnegative; fliplr in [0, 1]"
            )
        if "log" not in kwargs or "base_folder" not in kwargs:
            raise ValueError("Training requires log and base_folder directories")
        torch.manual_seed(int(kwargs.get("seed", 0)))
        self._device = self._select_device(kwargs.get("device", ""))
        if kwargs.get("model"):
            self.model = kwargs["model"]
        self._load(pretrained=kwargs.get("pretrained", True))
        configured = kwargs.get("classes") or getattr(self, "output", [])
        if isinstance(configured, str):
            configured = [configured]
        classes = sorted(
            set(
                configured
                or [
                    image.name
                    for image in images
                    if not kwargs.get("exclude") or image.partition != kwargs["exclude"]
                ]
            )
        )
        if not classes:
            raise ValueError("No classes provided/found for torchvision training")
        log = (Path(kwargs["base_folder"]) / kwargs["log"]).resolve()
        train_dir, val_dir = self._prepare_assets(
            classes,
            log,
            kwargs.get("validation", ""),
            kwargs.get("exclude", ""),
            *images,
            min_size=kwargs.get("min_size", 0),
        )
        # Retain a fine-tuned head only when its class order still matches exactly.
        if classes != self._categories:
            _replace_head(self.model, len(classes))
        self._categories = classes
        self.model.to(self._device)
        freeze = kwargs.get("freeze_backbone", False)
        self.model.requires_grad_(not freeze)
        if freeze:
            for name in ("fc", "classifier", "heads", "head"):
                head = getattr(self.model, name, None)
                if head is not None:
                    layers = [
                        m
                        for m in head.modules()
                        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
                    ]
                    if layers:
                        layers[-1].requires_grad_(True)
                        break
        train_data = ImageFolder(
            train_dir, transform=self._transforms(training=True, fliplr=fliplr)
        )
        val_data = ImageFolder(val_dir, transform=self._transforms(), allow_empty=True)
        train_loader = DataLoader(
            train_data, batch_size=batch, shuffle=True, num_workers=workers
        )
        val_loader = (
            DataLoader(val_data, batch_size=batch, num_workers=workers)
            if len(val_data)
            else None
        )
        if val_loader is None and kwargs.get("validation"):
            raise ValueError(
                f"No validation images match partition {kwargs['validation']!r} "
                "after exclusions and size filtering"
            )
        if val_loader is None:
            logger.warning(
                "No validation images; best checkpoint will use training loss, with early stopping disabled"
            )
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=float(kwargs.get("learning_rate", 0.0001)),
            weight_decay=float(kwargs.get("weight_decay", 0.0001)),
        )
        weights_dir = log / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "classes.json").write_text(
            json.dumps({"classes": classes}, indent=2)
        )
        history, best_loss, best_metrics, stale = [], math.inf, {}, 0
        for epoch in range(1, epochs + 1):
            self.model.train(
                not freeze
            )  # Frozen BatchNorm statistics must remain frozen too.
            train_metrics = self._epoch(train_loader, optimizer)
            self.model.eval()
            with torch.inference_mode():
                val_metrics = (
                    self._epoch(val_loader) if val_loader is not None else None
                )
            metrics = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
            metrics.update({f"val_{k}": v for k, v in (val_metrics or {}).items()})
            history.append(metrics)
            score = val_metrics["loss"] if val_metrics else train_metrics["loss"]
            if not math.isfinite(score):
                raise ValueError("Training produced a non-finite loss")
            checkpoint = self._checkpoint(epoch, metrics)
            torch.save(checkpoint, weights_dir / "last.pt")
            if score < best_loss:
                best_loss, best_metrics, stale = score, dict(metrics), 0
                torch.save(checkpoint, weights_dir / "best.pt")
            else:
                stale += 1
            logger.info("Epoch %d/%d: %s", epoch, epochs, metrics)
            with (log / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metrics))
                writer.writeheader()
                writer.writerows(history)
            if val_loader is not None and patience and stale >= patience:
                break
        self.original_model_path = weights_dir / "best.pt"
        self._reload()
        best_metrics.update(classes=classes, epochs_completed=len(history))
        (log / "metrics.json").write_text(json.dumps(best_metrics, indent=2))
        return ClassificationTrainResult(save_dir=log, results_dict=best_metrics)

    def _epoch(self, loader, optimizer=None) -> dict:
        import torch
        from torch.nn import functional as F

        total, correct, correct5, total_loss = 0, 0, 0, 0.0
        for pixels, labels in loader:
            pixels, labels = pixels.to(self._device), labels.to(self._device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = self.model(pixels)
            loss = F.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            correct5 += (
                (
                    logits.topk(min(5, len(self._categories)), dim=1).indices
                    == labels[:, None]
                )
                .any(1)
                .sum()
                .item()
            )
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "top5_accuracy": correct5 / total,
        }
