"""Torchvision image classification with portable, metadata-bearing checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cappa import Arg

from collectra.cli import command
from collectra.logger import get_logger
from collectra.types.images import Image, ImageCrop
from collectra.types.links import Link
from collectra.utils import threading_locked
from ..machine_learning.training import run_training_command
from .training import (
    ClassificationTrainResult,
    TorchClassifierTask,
    prepare_classification_inputs,
)

__all__ = ["ImageClassifierTorchvision"]
logger = get_logger(__name__)


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


class ImageClassifierTorchvision(TorchClassifierTask):
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

    def _transforms(
        self, *, training: bool = False, fliplr: float = 0.0, flipud: float = 0.0
    ):
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
        if training and flipud:
            steps.append(T.RandomVerticalFlip(flipud))
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
        wandb: Annotated[
            bool,
            Arg(help="Log training metrics to Weights & Biases."),
        ] = False,
        device: Annotated[
            str,
            Arg(
                help="Torch device such as cpu, cuda, cuda:0, or mps; omit for automatic selection."
            ),
        ] = "",
        seed: Annotated[
            int, Arg(help="Random seed for model initialization and data loading.")
        ] = 0,
        flipud: Annotated[
            float,
            Arg(help="Probability of vertical flipping during training, from 0 to 1."),
        ] = 0.0,
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
            flipud=flipud,
            workers=workers,
            wandb=wandb,
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

    def _prepare_training_model(self, classes, pretrained, freeze):
        import torch

        self._load(pretrained=pretrained)
        # Retain a fine-tuned head only when its class order still matches exactly.
        if classes != self._categories:
            _replace_head(self.model, len(classes))
        self._categories = classes
        self.model.to(self._device)
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

    def _forward(self, pixels):
        return self.model(pixels.to(self._device))
