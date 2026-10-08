"""Shared RGB YOLO backbone with attention fusion before the neck.

Custom model classes live in this importable module so native Ultralytics .pt
checkpoints survive a new Python process. Use load_adapted_model (or Collectra)
for training; ordinary YOLO reconstruction deliberately fails on the tagged
YAML instead of silently throwing away attention weights.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

FORMAT_VERSION = 1
RGB_CONV_TAG = "CollectraFeatureFusionRGBConv"


class ViewAttention(nn.Module):
    """Reference-conditioned, per-position softmax attention over RGB views."""

    def __init__(
        self, channels: int, num_views: int, reference_view: int, hidden_dim: int
    ):
        super().__init__()
        self.reference_view = reference_view
        self.score = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.view_bias = nn.Parameter(torch.zeros(num_views))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = features.shape
        reference = features[:, self.reference_view]
        query = reference[:, None].expand_as(features)
        paired = torch.cat((features, query), dim=2).reshape(
            batch * views, 2 * channels, height, width
        )
        logits = self.score(paired).reshape(batch, views, height, width)
        weights = (
            (logits + self.view_bias[None, :, None, None])
            .float()
            .softmax(1)
            .to(features.dtype)
        )
        pooled = (features * weights[:, :, None]).sum(1)
        return reference + self.gate.tanh() * (pooled - reference)


def _validate_spec(spec: dict) -> None:
    if spec.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported YOLO feature-fusion checkpoint version")
    views, reference, hidden = (
        spec["num_views"],
        spec["reference_view"],
        spec["hidden_dim"],
    )
    if views < 1 or not 0 <= reference < views or hidden < 1:
        raise ValueError(
            "Require num_views >= 1, 0 <= reference_view < num_views, hidden_dim >= 1"
        )


def _layer_input(layer, current, saved):
    if layer.f == -1:
        return current
    return (
        saved[layer.f]
        if isinstance(layer.f, int)
        else [current if index == -1 else saved[index] for index in layer.f]
    )


class FeatureFusionDetectionModel(DetectionModel):
    """Batched per-view backbone followed by fused multiscale neck inputs."""

    def __init__(self, cfg: dict, ch: int | None = None, nc=None, verbose=True):
        config = deepcopy(cfg)
        spec = config.pop("feature_fusion", None)
        if spec is None:
            raise ValueError("A feature-fusion model configuration is required")
        _validate_spec(spec)
        if ch is not None and ch != 3 * spec["num_views"]:
            raise ValueError(
                f"Dataset has {ch} channels; fusion model expects {3 * spec['num_views']}"
            )
        if config["backbone"][0][2] == RGB_CONV_TAG:
            config["backbone"][0][2] = "Conv"
        config["channels"] = 3
        # Stride initialization in DetectionModel calls _predict_once before
        # fusion is installed; its RGB path remains available during construction.
        super().__init__(cfg=config, ch=3, nc=nc, verbose=False)
        self._install_fusion(spec)
        if verbose:
            self.info()

    def _install_fusion(self, spec: dict) -> None:
        _validate_spec(spec)
        if hasattr(self, "view_fusion"):
            raise ValueError("Model is already adapted for feature fusion")
        first = self.model[0]
        if (
            self.yaml["backbone"][0][2] != "Conv"
            or not hasattr(first, "conv")
            or first.conv.in_channels != 3
        ):
            raise ValueError(
                "Expected an unmodified RGB YOLO detection backbone starting with Conv"
            )
        if not hasattr(first, "bn"):
            raise ValueError(
                "Use an unfused training checkpoint, not a Conv/BatchNorm-fused inference model"
            )
        self.num_views = spec["num_views"]
        self.backbone_layers = len(self.yaml["backbone"])
        # Fuse every saved backbone output plus its final output. This preserves
        # the full graph, including all backbone-to-neck skip connections.
        indices = sorted(
            {index for index in self.save if index < self.backbone_layers}
            | {self.backbone_layers - 1}
        )
        parameter = next(self.parameters())
        flags = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            size = max(64, 2 * int(self.stride.max()))
            current = torch.zeros(
                1, 3, size, size, device=parameter.device, dtype=parameter.dtype
            )
            saved, channels = [], {}
            with torch.no_grad():
                for layer in self.model[: self.backbone_layers]:
                    current = layer(_layer_input(layer, current, saved))
                    if not isinstance(current, torch.Tensor) or current.ndim != 4:
                        raise ValueError(
                            "Only tensor feature maps from standard YOLO detection backbones are supported"
                        )
                    saved.append(current if layer.i in self.save else None)
                    if layer.i in indices:
                        channels[str(layer.i)] = current.shape[1]
        finally:
            for module, training in flags:
                module.training = training
        self.view_fusion = nn.ModuleDict(
            {
                index: ViewAttention(
                    width, self.num_views, spec["reference_view"], spec["hidden_dim"]
                )
                for index, width in channels.items()
            }
        ).to(device=parameter.device, dtype=parameter.dtype)
        self.view_fusion.train(self.training)
        self.yaml = deepcopy(self.yaml)
        self.yaml["channels"] = 3 * self.num_views
        self.yaml["feature_fusion"] = dict(spec)
        # A vanilla trainer must not rebuild a widened RGB model and silently
        # discard fusion. Our trainer removes this marker before RGB construction.
        self.yaml["backbone"][0][2] = RGB_CONV_TAG
        self.task = "detect"

    def _fuse_feature(
        self, index: int, feature: torch.Tensor, batch: int
    ) -> torch.Tensor:
        return self.view_fusion[str(index)](
            feature.reshape(batch, self.num_views, *feature.shape[1:])
        )

    def _predict_once(self, x, profile=False, embed=None):
        if not hasattr(self, "view_fusion"):
            return super()._predict_once(x, profile=profile, embed=embed)
        if profile or embed:
            raise NotImplementedError(
                "Feature-fusion layer profiling/embedding extraction is not supported"
            )
        if x.ndim != 4 or x.shape[1] != 3 * self.num_views:
            raise ValueError(
                f"Expected BCHW with {3 * self.num_views} channels; got {tuple(x.shape)}"
            )
        batch, _, height, width = x.shape
        current = x.reshape(batch * self.num_views, 3, height, width)
        saved = []
        for layer in self.model:
            if layer.i == self.backbone_layers:
                last = self.backbone_layers - 1
                for index in self.view_fusion:
                    i = int(index)
                    if saved[i] is not None:
                        saved[i] = self._fuse_feature(i, saved[i], batch)
                current = (
                    saved[last]
                    if saved[last] is not None
                    else self._fuse_feature(last, current, batch)
                )
            current = layer(_layer_input(layer, current, saved))
            saved.append(current if layer.i in self.save else None)
        return current

    def load(self, weights, verbose=True):
        source = (
            (weights.get("ema") or weights["model"])
            if isinstance(weights, dict)
            else weights
        )
        if not isinstance(source, FeatureFusionDetectionModel):
            raise ValueError(
                "Load an RGB checkpoint through adapt_model first; training requires fusion weights"
            )
        if source.yaml["feature_fusion"] != self.yaml["feature_fusion"]:
            raise ValueError("Source and destination fusion configurations differ")
        # Class-count changes are handled by Ultralytics; fusion weights must all transfer.
        expected = {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith("view_fusion.")
        }
        state = source.state_dict()
        if any(
            key not in state or state[key].shape != value.shape
            for key, value in expected.items()
        ):
            raise ValueError("Checkpoint has missing or incompatible attention weights")
        return super().load(weights, verbose=verbose)


class FeatureFusionDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if not isinstance(cfg, dict) or "feature_fusion" not in cfg:
            raise ValueError("Feature-fusion training requires an adapted checkpoint")
        model = self.set_model_names_for_load(
            FeatureFusionDetectionModel(
                cfg, ch=self.data["channels"], nc=self.data["nc"], verbose=verbose
            )
        )
        if weights is not None:
            model.load(weights, verbose=verbose)
        return model

    def plot_training_samples(self, batch, ni):
        """RGB training plots cannot represent concatenated RGB views."""


class FeatureFusionYOLO(YOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping["detect"]["model"] = FeatureFusionDetectionModel
        mapping["detect"]["trainer"] = FeatureFusionDetectionTrainer
        return mapping

    def train(self, trainer=None, **kwargs):
        if trainer is not None and trainer is not FeatureFusionDetectionTrainer:
            raise ValueError(
                "Use FeatureFusionDetectionTrainer to preserve the fusion architecture"
            )
        device = kwargs.get("device", self.overrides.get("device"))
        if isinstance(device, (list, tuple)) or (
            isinstance(device, str) and "," in device
        ):
            raise ValueError(
                "Feature fusion currently supports single-device training only"
            )
        first = self.model.model[0]
        if not hasattr(first, "bn"):
            raise ValueError(
                "Prediction fused BatchNorm; reload the unfused checkpoint before training"
            )
        kwargs.setdefault("plots", False)
        return super().train(trainer=FeatureFusionDetectionTrainer, **kwargs)

    def export(self, **kwargs):
        raise NotImplementedError(
            "YOLO feature fusion currently supports native PyTorch, not exported runtimes"
        )


def enable_feature_fusion(yolo):
    """Upgrade a loaded YOLO wrapper so training reconstructs its custom model."""
    if isinstance(yolo.model, FeatureFusionDetectionModel):
        _validate_spec(yolo.model.yaml["feature_fusion"])
        if type(yolo) is YOLO:
            yolo.__class__ = FeatureFusionYOLO
        elif not isinstance(yolo, FeatureFusionYOLO):
            raise ValueError("Only standard YOLO detection wrappers are supported")
    return yolo


def adapt_model(yolo, num_views: int, reference_view: int = 0, hidden_dim: int = 64):
    spec = dict(
        version=FORMAT_VERSION,
        num_views=num_views,
        reference_view=reference_view,
        hidden_dim=hidden_dim,
    )
    _validate_spec(spec)
    if (
        yolo.task != "detect"
        or type(yolo.model) is not DetectionModel
        or type(yolo) is not YOLO
    ):
        raise ValueError(
            "Expected a standard, unmodified Ultralytics YOLO RGB detection model"
        )
    yolo.model.__class__ = FeatureFusionDetectionModel
    yolo.model._install_fusion(spec)
    return enable_feature_fusion(yolo)


def save_adapted_model(yolo, path: str | Path) -> None:
    if not isinstance(yolo.model, FeatureFusionDetectionModel):
        raise ValueError("Model is not adapted for YOLO feature fusion")
    path = Path(path)
    if path.suffix.lower() != ".pt":
        raise ValueError(
            "Native Ultralytics feature-fusion checkpoints must use a .pt suffix"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = yolo.ckpt
    try:
        yolo.ckpt = previous or {}
        yolo.save(str(path))
    finally:
        yolo.ckpt = previous


def load_adapted_model(path: str | Path):
    yolo = YOLO(str(path))
    if not isinstance(yolo.model, FeatureFusionDetectionModel):
        raise ValueError("Checkpoint is not a YOLO feature-fusion model")
    return enable_feature_fusion(yolo)
