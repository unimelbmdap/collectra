from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, TYPE_CHECKING, Any

from pydantic import Field

from cappa import Arg

from collectra.cli import command
from .adaptation_commands import RFDETRAdaptationCommands
from .base import (
    Batch,
    Epochs,
    LearningRate,
    ObjectDetector,
    RunResolution,
    Threshold,
    TrainResolution,
    WeightDecay,
)
from collectra.types.images import (
    Image,
    ImageCrop,
    read_tiff_channels,
)
from collectra.utils import change_dir

from ...logger import get_logger
from ..base import Task
from collectra.utils import threading_locked
from ..machine_learning.class_names import (
    read_class_names,
    sidecar_candidates,
    write_class_names,
)
from ..machine_learning.training import (
    prepare_object_detection_inputs,
    run_training_command,
)
from .coco_data import CocoDetectionData
from .detr import DetectionTrainResult

if TYPE_CHECKING:
    from rfdetr.detr import RFDETR

logger = get_logger(__name__)

__all__ = ["ObjectDetectorRFDETR", "ObjectDetectionRFDETR"]

_MODEL_VARIANTS = {
    "base": "RFDETRBase",
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}


def __getattr__(name: str):
    """Expose patchable backend classes without importing them during CLI help."""
    if name == "RFDETR":
        from rfdetr.detr import RFDETR

        return RFDETR
    if name in _MODEL_VARIANTS.values():
        import rfdetr

        return getattr(rfdetr, name)
    raise AttributeError(name)


def _backend_class(name: str):
    return globals().get(name) or __getattr__(name)


def class_head_size(model) -> int | None:
    """The number of outputs of the model's last classification layer, if it has one.

    RF-DETR gives that layer one output for each class plus one for the background.
    """
    head = getattr(
        getattr(getattr(model, "model", None), "model", None), "class_embed", None
    )
    return getattr(head, "out_features", None)


def load_rfdetr_model(model_path: str | Path) -> RFDETR:
    from .rfdetr_channels import (
        absolute_checkpoint,
        multichannel_checkpoints,
        sync_channels,
    )

    model_path = absolute_checkpoint(model_path)

    # Fusion checkpoints retain a three-channel projection but consume many
    # RGB views. Ordinary RF-DETR loading would silently discard fusion weights.
    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = {}
    if "backbone.0.encoder.fusion_spec" in checkpoint.get("model", {}):
        from .rfdetr_feature_fusion import load_adapted_model

        return load_adapted_model(model_path)
    del checkpoint

    errors = []
    try:
        # Resolves the model size from checkpoint metadata. The loader widens
        # the input layer to the stored channel count; num_channels=3 stops
        # RF-DETR from widening it a second time.
        with multichannel_checkpoints():
            model = _backend_class("RFDETR").from_checkpoint(
                str(model_path), num_channels=3
            )
        if isinstance(model, _backend_class("RFDETR")):
            sync_channels(model)
            return model
    except Exception as e:
        errors.append(("RFDETR.from_checkpoint", str(e)))

    # Fallback for checkpoints without metadata. RF-DETR loads non-strictly, so
    # a smaller variant can "load" a larger checkpoint while dropping weights.
    for class_name in _MODEL_VARIANTS.values():
        try:
            model_class = _backend_class(class_name)
            with multichannel_checkpoints():
                model = model_class(pretrain_weights=str(model_path))
            sync_channels(model)
            return model
        except Exception as e:
            errors.append((class_name, str(e)))

    logger.error("Failed to load RF-DETR model from %s. Errors: %s", model_path, errors)


class ObjectDetectorRFDETR(CocoDetectionData, RFDETRAdaptationCommands, ObjectDetector):
    """RF-DETR task backed by the Roboflow rfdetr package."""

    model: Any = None  # a path or a loaded RF-DETR model
    original_model_path: str | Path | None = None
    batch_size: int = Field(default=1, ge=1)

    def __init__(
        self,
        name: str,
        model: str | Path | object | None = None,
        **kwargs,
    ):
        super().__init__(name, model=model, **kwargs)
        self._device = None
        self._categories: list[str] = []

    def _init_model(self) -> None:
        self._load()

        if not isinstance(self.model, _backend_class("RFDETR")):
            raise ValueError("Model must be an RFDETR instance")

    def _reload(self) -> None:
        if not self.original_model_path:
            logger.warning("Original model path is not set. Skipping...")
            return

        self.model = load_rfdetr_model(self.original_model_path)
        self._set_categories(Path(self.original_model_path))

    def _load_categories_from_sidecar(self, checkpoint_path: Path) -> None:
        classes = read_class_names(checkpoint_path)
        if classes is not None:
            self._categories = classes

    def _output_names(self) -> list[str]:
        """Names of this task's output nodes, in the order the pipeline lists them."""
        pipeline = self.pipeline
        if pipeline is None or not pipeline.node_manager.has_node(self.name):
            return []
        node = pipeline.node_manager.resolve_node(self.name)
        return [str(output.name) for output in pipeline._output_nodes(node)]

    def _set_categories(self, checkpoint_path: Path) -> None:
        """Name the model's classes: the pipeline's outputs, else the sidecar.

        The pipeline is the source of truth. A sidecar that disagrees is warned
        about and ignored. Either way the last layer must fit the names.
        """
        self._load_categories_from_sidecar(checkpoint_path)
        outputs = self._output_names()
        if outputs:
            if self._categories and self._categories != outputs:
                logger.warning(
                    "Class names in the sidecar next to %s (%s) differ from the outputs "
                    "of task %r (%s); using the pipeline's outputs.",
                    checkpoint_path,
                    ", ".join(self._categories),
                    self.name,
                    ", ".join(outputs),
                )
            self._categories = outputs
        size = class_head_size(self.model)
        if self._categories and size is not None and size != len(self._categories) + 1:
            raise ValueError(
                f"Task {self.name!r} has {len(self._categories)} output(s) "
                f"({', '.join(self._categories)}) but the model has {size - 1} classes."
            )

    def _load(self) -> None:
        if isinstance(self.model, _backend_class("RFDETR")):
            return

        self._device = self._select_device()

        variant = "base" if self.model is None else None
        if isinstance(self.model, str):
            name = self.model.lower()
            if name == "default":
                variant = "base"
            elif name in _MODEL_VARIANTS:
                variant = name
            else:
                variant = next(
                    (
                        key
                        for key, value in _MODEL_VARIANTS.items()
                        if value.lower() == name
                    ),
                    None,
                )
        if variant is not None:
            self.model = _backend_class(_MODEL_VARIANTS[variant])()
            self.original_model_path = None
            from rfdetr.assets.coco_classes import COCO_CLASSES

            self._categories = list(COCO_CLASSES)
            return

        if not isinstance(self.model, (str, Path)):
            raise ValueError(
                "Model must be None, 'default', a model variant name, "
                "a checkpoint path, or an RFDETR instance"
            )

        checkpoint_path = Path(str(self.model))
        if checkpoint_path.exists() and checkpoint_path.is_file():
            self.original_model_path = checkpoint_path
            self.model = load_rfdetr_model(self.original_model_path)
            self._categories = []
            self._set_categories(checkpoint_path)
            candidates = sidecar_candidates(checkpoint_path)
            if not self._categories and not any(c.exists() for c in candidates):
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
            "RF-DETR requires a valid checkpoint path, None/'default', "
            f"or a model variant: {', '.join(_MODEL_VARIANTS)}."
        )

    def _tiff_tensor(self, image: Image):
        """Read every channel of a TIFF image as a CHW float tensor in [0, 1]."""
        import numpy as np
        import torch

        from .rfdetr_data import scale_tiff_pixels

        pixels = read_tiff_channels(image.get_path())
        if isinstance(image, ImageCrop):
            left, top, right, bottom = image.coordinates()
            if right <= left or bottom <= top:
                raise ValueError(f"ImageCrop {image.id!r} produces an empty pixel crop")
            pixels = pixels[top:bottom, left:right, :]
        pixels = np.rot90(pixels, k=image.orientation.to_degree() // 90)

        expected = self.model.model_config.num_channels
        if pixels.shape[2] != expected:
            raise ValueError(
                f"{image.get_path()}: {pixels.shape[2]} image channels, "
                f"but the detection model expects {expected}"
            )
        pixels = scale_tiff_pixels(pixels)
        return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

    @command(name="run")
    def cli_run(
        self,
        inputs: list[str],
        output: Path | None = None,
        verbose: bool = False,
        usage: bool = False,
        render: bool = False,
        model: Annotated[
            Path | None,
            Arg(help="Model checkpoint path. Omit to keep the configured model."),
        ] = None,
        resolution: RunResolution = None,
        threshold: Threshold = None,
        batch_size: Annotated[
            int,
            Arg(
                help="Number of files to run through the model at once; results are saved after each batch."
            ),
        ] = 16,
        force: Annotated[
            bool,
            Arg(
                help="Rerun tasks whose outputs already exist, replacing those outputs. By default such tasks are skipped."
            ),
        ] = False,
    ) -> None:
        """Run this object detector for the supplied input files."""
        if batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        if model is not None:
            self.model = model
        if resolution is not None:
            self.resolution = resolution
        if threshold is not None:
            self.threshold = threshold
        self.batch_size = batch_size
        super().cli_run(
            inputs,
            output=output,
            verbose=verbose,
            usage=usage,
            render=render,
            force=force,
        )

    def _predict(self, images: list[Image]) -> list[tuple]:
        """Run the model over ``images`` together as ``(detections, width, height)``."""
        threshold = self.threshold
        inputs = []
        sizes = []
        for image in images:
            if image.get_path().suffix.lower() in {".tif", ".tiff"}:
                model_input = self._tiff_tensor(image)
                height, width = model_input.shape[1:]
            else:
                model_input = image.pil().convert("RGB")
                width, height = model_input.size
            inputs.append(model_input)
            sizes.append((width, height))

        kwargs = {}
        resolution = self.inference_resolution
        if resolution is not None:
            kwargs["shape"] = (int(resolution), int(resolution))
        detections = self.model.predict(
            inputs if len(inputs) > 1 else inputs[0],
            threshold=threshold,
            include_source_image=False,
            **kwargs,
        )
        if not isinstance(detections, list):
            detections = [detections]
        return [(d, w, h) for d, (w, h) in zip(detections, sizes)]

    def _predict_batch(self, images: list[Image]) -> list[tuple]:
        return self._predict(images)

    def _before_prefetch(self) -> None:
        logger.info(
            "RF-DETR inference resolution: %s",
            self.inference_resolution
            or getattr(self.model.model_config, "resolution", "model default"),
        )

    @threading_locked()
    def run(self, *args: Image) -> list[ImageCrop]:
        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        if not isinstance(args[0], Image):
            raise TypeError("Input must be an instance of Image.")

        image = args[0]
        self._init_model()

        predicted = self._take_prefetched(image)
        detections, width, height = predicted or self._predict([image])[0]

        confidences = getattr(detections, "confidence", None)
        results: list[ImageCrop] = []
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
                    confidence=(
                        float(confidences[i]) if confidences is not None else None
                    ),
                    orientation=image.orientation,
                    name=class_name,
                )
            )

        print(f"Found {len(results)} objects in the image.")
        return results

    def _prepare_params(self, **kwargs) -> dict:
        params = {
            "epochs": int(kwargs.get("epochs", 1)),
            "batch_size": int(kwargs.get("batch", 4)),
            "grad_accum_steps": int(kwargs.get("grad_accum_steps", 4)),
            "lr": float(kwargs.get("lr", 1e-4)),
            "lr_encoder": float(kwargs.get("lr_encoder", 1.5e-4)),
            "lr_vit_layer_decay": float(kwargs.get("lr_vit_layer_decay", 0.8)),
            "lr_component_decay": float(kwargs.get("lr_component_decay", 0.7)),
            "warmup_epochs": float(kwargs.get("warmup_epochs", 0.0)),
            "weight_decay": float(kwargs.get("weight_decay", 1e-4)),
            "drop_path": float(kwargs.get("drop_path", 0.0)),
            "multi_scale": bool(kwargs.get("multi_scale", True)),
            "expanded_scales": bool(kwargs.get("expanded_scales", True)),
            "do_random_resize_via_padding": bool(
                kwargs.get("do_random_resize_via_padding", False)
            ),
            "augmentation_backend": kwargs.get("augmentation_backend", "cpu"),
            "use_ema": bool(kwargs.get("use_ema", True)),
            "ema_decay": float(kwargs.get("ema_decay", 0.993)),
            "ema_tau": int(kwargs.get("ema_tau", 100)),
            "ema_update_interval": int(kwargs.get("ema_update_interval", 1)),
            "output_dir": str(kwargs["output_dir"]),
            "wandb": bool(kwargs.get("wandb", False)),
            "project": kwargs.get("project", "runs/rfdetr"),
            "run": kwargs.get("log", "rfdetr-run"),
            "early_stopping": bool(kwargs.get("early_stopping", False)),
            "early_stopping_patience": int(kwargs.get("early_stopping_patience", 10)),
            "early_stopping_min_delta": float(
                kwargs.get("early_stopping_min_delta", 0.001)
            ),
            "early_stopping_use_ema": bool(kwargs.get("early_stopping_use_ema", False)),
        }

        resolution = kwargs.get("resolution")
        if resolution is not None:
            params["resolution"] = int(resolution)

        augmentation = kwargs.get("augmentation", "default").lower()
        if augmentation != "default":
            preset_names = {
                "none",
                "conservative",
                "aggressive",
                "aerial",
                "industrial",
            }
            if augmentation not in preset_names:
                raise ValueError(
                    "augmentation must be one of: default, none, conservative, "
                    "aggressive, aerial, industrial"
                )
            from rfdetr.datasets.aug_configs import (
                AUG_AERIAL,
                AUG_AGGRESSIVE,
                AUG_CONSERVATIVE,
                AUG_INDUSTRIAL,
            )

            presets = {
                "none": {},
                "conservative": AUG_CONSERVATIVE,
                "aggressive": AUG_AGGRESSIVE,
                "aerial": AUG_AERIAL,
                "industrial": AUG_INDUSTRIAL,
            }
            params["aug_config"] = presets[augmentation]
        return params

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
        self._write_coco_dataset(
            train_samples,
            val_samples,
            classes,
            dataset_dir,
            rgb=kwargs.get("rgb", False),
        )

        params_kwargs = dict(kwargs)
        params_kwargs["log"] = f"{log.name}"
        params_kwargs["output_dir"] = str(weights_dir)
        params = self._prepare_params(**params_kwargs)
        has_tiffs = any(
            Path(sample["image_path"]).suffix.lower() in {".tif", ".tiff"}
            for sample in [*train_samples, *val_samples]
        )
        if has_tiffs:
            from .rfdetr_channels import require_multichannel_rfdetr

            require_multichannel_rfdetr()
        if has_tiffs and params["augmentation_backend"] != "torchvision":
            # Albumentations round-trips through PIL; the GPU path uses
            # three-channel normalization outside our dataset patch. "auto"
            # can select either path, so TIFFs must explicitly use torchvision.
            logger.info("TIFF inputs: using the torchvision augmentation backend")
            params["augmentation_backend"] = "torchvision"

        from .rfdetr_channels import multichannel_checkpoints
        from .rfdetr_data import multichannel_datasets
        from .rfdetr_curves import wandb_detection_curves

        model = self.model
        with (
            multichannel_datasets(),
            multichannel_checkpoints(),
            wandb_detection_curves(params["wandb"] and bool(val_samples), log, classes),
        ):
            model.train(
                dataset_dir=str(dataset_dir),
                **params,
            )

        self._categories = classes
        self.model = model

        # Persist class names alongside checkpoint for later reloading
        write_class_names(weights_dir / "classes.yaml", classes)

        # RF-DETR's total checkpoint selects the validation winner across
        # regular and EMA weights. Prefer it so best.pt keeps that winner.
        best_pt = weights_dir / "best.pt"
        for candidate in [
            weights_dir / "checkpoint_best_total.pth",
            weights_dir / "checkpoint_best_regular.pth",
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
            Arg(help="Partition value identifying validation examples."),
        ] = "",
        exclude: Annotated[
            str,
            Arg(help="Partition value identifying examples to exclude from training."),
        ] = "",
        epochs: Epochs = None,
        batch: Batch = None,
        grad_accum_steps: Annotated[
            int,
            Arg(
                help="Accumulate gradients over this many batches before an optimizer step."
            ),
        ] = 4,
        learning_rate: LearningRate = None,
        rgb: Annotated[
            bool,
            Arg(
                help="Convert all prepared training images to 3-channel RGB, so images with differing channel counts (grayscale, RGBA) can be trained together."
            ),
        ] = False,
        wandb: Annotated[
            bool,
            Arg(help="Log training metrics to Weights & Biases."),
        ] = False,
        early_stopping: Annotated[
            bool,
            Arg(help="Stop training when validation performance stops improving."),
        ] = False,
        weight_decay: WeightDecay = None,
        drop_path: Annotated[
            float,
            Arg(help="Stochastic depth drop probability used for regularization."),
        ] = 0.0,
        augmentation: Annotated[
            str,
            Arg(
                help="Augmentation preset: default, none, conservative, aggressive, aerial, or industrial."
            ),
        ] = "default",
        augmentation_backend: Annotated[
            str,
            Arg(help="Augmentation execution backend: cpu, auto, or gpu."),
        ] = "cpu",
        multi_scale: Annotated[
            bool,
            Arg(help="Train with multiple image resolutions."),
        ] = True,
        expanded_scales: Annotated[
            bool,
            Arg(help="Use an expanded range of resolutions for multi-scale training."),
        ] = True,
        do_random_resize_via_padding: Annotated[
            bool,
            Arg(help="Use padding for random image resizing."),
        ] = False,
        use_ema: Annotated[
            bool,
            Arg(help="Maintain an exponential moving average (EMA) of model weights."),
        ] = True,
        ema_decay: Annotated[
            float,
            Arg(
                help="Decay factor for the exponential moving average of model weights."
            ),
        ] = 0.993,
        ema_tau: Annotated[
            int,
            Arg(help="Time constant controlling the warmup of EMA decay."),
        ] = 100,
        ema_update_interval: Annotated[
            int,
            Arg(help="Number of optimizer steps between EMA updates."),
        ] = 1,
        early_stopping_patience: Annotated[
            int,
            Arg(
                help="Number of validation checks without sufficient improvement before stopping."
            ),
        ] = 10,
        early_stopping_min_delta: Annotated[
            float,
            Arg(help="Minimum improvement required to reset early-stopping patience."),
        ] = 0.001,
        early_stopping_use_ema: Annotated[
            bool,
            Arg(help="Use EMA model metrics for early stopping."),
        ] = False,
        lr_encoder: Annotated[
            float,
            Arg(help="Learning rate for the image encoder."),
        ] = 1.5e-4,
        lr_vit_layer_decay: Annotated[
            float,
            Arg(
                help="Layer-wise learning-rate decay factor for the vision transformer."
            ),
        ] = 0.8,
        lr_component_decay: Annotated[
            float,
            Arg(help="Learning-rate decay factor for encoder components."),
        ] = 0.7,
        warmup_epochs: Annotated[
            float,
            Arg(
                help="Number of epochs for learning-rate warmup; fractional values are allowed."
            ),
        ] = 0.0,
        model: Annotated[
            str,
            Arg(
                help="Checkpoint path or pretrained variant: nano, small, base, medium, large, xlarge, or 2xlarge (full RFDETR class names also accepted). XL variants require rfdetr[plus]. 'default' selects Base; omit to keep the configured model."
            ),
        ] = "",
        resolution: TrainResolution = None,
        max_items: Annotated[
            int,
            Arg(
                help="Use at most this many images in each of the training and validation sets (0 uses all); for quick tests."
            ),
        ] = 0,
    ):
        """Train this RF-DETR object detector."""
        settings = self._settings_for_run(
            epochs=epochs,
            batch=batch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            resolution=resolution,
        )
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
            grad_accum_steps=grad_accum_steps,
            lr=settings["learning_rate"],
            rgb=rgb,
            wandb=wandb,
            early_stopping=early_stopping,
            weight_decay=settings["weight_decay"],
            drop_path=drop_path,
            augmentation=augmentation,
            augmentation_backend=augmentation_backend,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            do_random_resize_via_padding=do_random_resize_via_padding,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_tau=ema_tau,
            ema_update_interval=ema_update_interval,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_use_ema=early_stopping_use_ema,
            lr_encoder=lr_encoder,
            lr_vit_layer_decay=lr_vit_layer_decay,
            lr_component_decay=lr_component_decay,
            warmup_epochs=warmup_epochs,
            model=model,
            resolution=settings["resolution"],
            max_items=max_items,
        )

    def _train(self, *images: ImageCrop, **kwargs) -> DetectionTrainResult:
        if kwargs.get("model"):
            self.model = kwargs["model"]

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
        max_items = int(kwargs.get("max_items", 0) or 0)
        if max_items > 0:
            train_samples = train_samples[:max_items]
            val_samples = val_samples[:max_items]
            print(
                f"Limiting to {len(train_samples)} training and "
                f"{len(val_samples)} validation images (max_items={max_items})"
            )
        self._check_distribution(classes, train_samples, val_samples)

        with change_dir(kwargs["base_folder"]):
            results = self._train_fold(train_samples, val_samples, classes, log, kwargs)

        print(
            f"Best RF-DETR model saved to: {results.save_dir / 'weights' / 'best.pt'}"
        )
        return results


# The former name of ObjectDetectorRFDETR.
ObjectDetectionRFDETR = ObjectDetectorRFDETR
