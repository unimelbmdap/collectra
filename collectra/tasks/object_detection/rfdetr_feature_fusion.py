"""Shared RGB DINOv2 encoding and attention pooling across aligned RGB views.

Checkpoints use state dictionaries, never serialized Python model objects.
The fusion specification is a persistent buffer, so upstream training/EMA
checkpoints remain reconstructible even when custom top-level metadata is lost.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import tempfile
import types

import torch
from torch import nn
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import DinoV2

SPEC_KEY = "backbone.0.encoder.fusion_spec"
FORMAT_VERSION = 1


class ViewAttention(nn.Module):
    """Reference-conditioned spatial attention over views, not over image pixels."""

    def __init__(
        self, channels: int, num_views: int, reference_view: int, hidden_dim: int
    ):
        super().__init__()
        self.reference_view = reference_view
        self.score = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        # Begin with uniform attention; the residual gate preserves RGB exactly.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.view_bias = nn.Parameter(torch.zeros(num_views))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        # B,V,C,H,W -> B,H,W,V,C. Attention is local to aligned positions.
        features = views.permute(0, 3, 4, 1, 2)
        reference = features[..., self.reference_view, :]
        query = reference.unsqueeze(-2).expand_as(features)
        logits = self.score(torch.cat((features, query), dim=-1)).squeeze(-1)
        weights = (logits + self.view_bias).float().softmax(dim=-1).to(features.dtype)
        pooled = (features * weights.unsqueeze(-1)).sum(dim=-2)
        fused = reference + self.gate.tanh() * (pooled - reference)
        return fused.permute(0, 3, 1, 2).contiguous()


class MultiViewDinoV2(DinoV2):
    def forward(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        if pixels.ndim != 4 or pixels.shape[1] != 3 * self.num_views:
            raise ValueError(
                f"Expected BCHW with {3 * self.num_views} channels; got {tuple(pixels.shape)}"
            )
        batch, _, height, width = pixels.shape
        # One encoder invocation batches all views; parameters are shared.
        rgb = pixels.reshape(batch * self.num_views, 3, height, width)
        features = super().forward(rgb)
        if len(features) != len(self.view_fusion):
            raise RuntimeError(
                "Encoder feature levels changed; cannot apply saved fusion modules"
            )
        return [
            fusion(feature.reshape(batch, self.num_views, *feature.shape[1:]))
            for feature, fusion in zip(features, self.view_fusion)
        ]

    def export(self) -> None:
        raise NotImplementedError(
            "Feature fusion currently supports eager prediction/training, not ONNX export"
        )


class FeatureFusionBackbone(Backbone):
    def get_named_param_lr_pairs(self, args, prefix="backbone.0"):
        pairs = super().get_named_param_lr_pairs(args, prefix)
        # New fusion parameters must not inherit the tiny pretrained embedding LR.
        for name, param in self.encoder.view_fusion.named_parameters():
            pairs[f"{prefix}.encoder.view_fusion.{name}"] = {
                "params": param,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        return pairs


def fusion_spec(state: dict) -> dict | None:
    value = state.get(SPEC_KEY)
    if value is None:
        return None
    version, views, reference, hidden = map(int, value.tolist())
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported feature-fusion checkpoint version: {version}")
    if views < 1 or not 0 <= reference < views or hidden < 1:
        raise ValueError("Invalid feature-fusion checkpoint specification")
    return {"num_views": views, "reference_view": reference, "hidden_dim": hidden}


def install_feature_fusion(
    nn_model, num_views: int, reference_view: int = 0, hidden_dim: int = 64
) -> None:
    if num_views < 1 or not 0 <= reference_view < num_views or hidden_dim < 1:
        raise ValueError(
            "Require num_views >= 1, 0 <= reference_view < num_views, hidden_dim >= 1"
        )
    backbone = nn_model.backbone[0]
    encoder = backbone.encoder
    if type(backbone) is not Backbone or type(encoder) is not DinoV2:
        raise ValueError(
            "Expected an unmodified RF-DETR DINOv2 backbone (LoRA/export/fusion wrappers are unsupported)"
        )
    projection = encoder.encoder.embeddings.patch_embeddings.projection
    if projection.in_channels != 3:
        raise ValueError(
            "Feature fusion requires an RGB checkpoint, not an expanded input projection"
        )
    # Preserve every existing weight and key; only add fusion parameters/buffer.
    encoder.__class__ = MultiViewDinoV2
    backbone.__class__ = FeatureFusionBackbone
    encoder.num_views = num_views
    encoder.view_fusion = nn.ModuleList(
        [
            ViewAttention(channels, num_views, reference_view, hidden_dim)
            for channels in encoder._out_feature_channels
        ]
    ).to(device=projection.weight.device, dtype=projection.weight.dtype)
    encoder.register_buffer(
        "fusion_spec",
        torch.tensor(
            [FORMAT_VERSION, num_views, reference_view, hidden_dim],
            dtype=torch.int64,
            device=projection.weight.device,
        ),
    )
    encoder.view_fusion.train(encoder.training)


def _sync(detector, spec: dict) -> None:
    channels = 3 * spec["num_views"]
    detector.model_config.num_channels = channels
    detector.model.args.num_channels = channels
    detector.means = [0.485, 0.456, 0.406] * spec["num_views"]
    detector.stds = [0.229, 0.224, 0.225] * spec["num_views"]
    detector.train = types.MethodType(_train_feature_fusion, detector)


def adapt_model(
    detector, num_views: int, reference_view: int = 0, hidden_dim: int = 64
):
    if detector.model_config.backbone_lora:
        raise ValueError("LoRA backbones are not supported by feature fusion")
    spec = dict(
        num_views=num_views, reference_view=reference_view, hidden_dim=hidden_dim
    )
    install_feature_fusion(detector.model.model, **spec)
    _sync(detector, spec)
    return detector


def save_adapted_model(detector, path: str | Path) -> None:
    state = detector.model.model.state_dict()
    spec = fusion_spec(state)
    if spec is None:
        raise ValueError("Model has no feature-fusion specification")
    config = detector.model_config.model_dump(mode="json")
    config["model_name"] = type(detector).__name__
    args = dict(config, class_names=list(detector.model.class_names or []))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "model_name": type(detector).__name__,
            "model_config": config,
            "args": args,
            "feature_fusion": spec,
        },
        path,
    )


def load_adapted_model(path: str | Path, device: str = "cpu"):
    import rfdetr

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    spec = fusion_spec(state)
    if spec is None:
        raise ValueError("Checkpoint is not an RF-DETR feature-fusion model")
    args = checkpoint.get("args", {})
    notes = args.get("notes") or {}
    saved_config = checkpoint.get("model_config")
    if saved_config is None and isinstance(notes, dict):
        saved_config = notes.get("collectra_feature_fusion", {}).get("model_config")
    if saved_config is None:
        raise ValueError("Fusion checkpoint has no reconstruction config")
    config = dict(saved_config)
    name = checkpoint.get("model_name") or config.get("model_name")
    if not isinstance(name, str) or not name.startswith("RFDETR"):
        raise ValueError("Checkpoint has no RFDETR model class")
    # Reconstruct RGB architecture before installing fusion. Never tile input weights.
    config.update(pretrain_weights=None, num_channels=3, device=device)
    config["num_classes"] = state["class_embed.weight"].shape[0] - 1
    detector = getattr(rfdetr, name)(**config)
    adapt_model(detector, **spec)
    detector.model.model.load_state_dict(state, strict=True)
    detector.model.class_names = checkpoint.get("args", {}).get("class_names", [])
    detector.model_config.pretrain_weights = str(Path(path).resolve())
    return detector


@contextlib.contextmanager
def feature_fusion_training(spec: dict):
    """Install fusion before upstream loads weights and creates the optimizer."""
    import rfdetr.training.module_model as module_model

    original = module_model.build_model_from_config

    def build(model_config, *args, **kwargs):
        model = original(model_config, *args, **kwargs)
        install_feature_fusion(model, **spec)
        # Upstream strips model_config from best_total. TrainConfig.notes lives
        # inside args and survives stripping, including custom resolutions.
        train_config = args[0] if args else kwargs.get("train_config")
        if train_config is not None:
            previous_notes = train_config.notes
            notes = (
                dict(previous_notes)
                if isinstance(previous_notes, dict)
                else (
                    {"user_notes": previous_notes} if previous_notes is not None else {}
                )
            )
            notes["collectra_feature_fusion"] = {
                "model_config": model_config.model_dump(mode="json"),
                **spec,
            }
            train_config.notes = notes
        return model

    module_model.build_model_from_config = build
    try:
        yield
    finally:
        module_model.build_model_from_config = original


def _train_feature_fusion(detector, **kwargs):
    """Train the current fusion weights rather than an unmodified RGB rebuild."""
    from rfdetr.detr import RFDETR
    from .rfdetr_data import multichannel_datasets
    from .rfdetr_channels import require_multichannel_rfdetr

    require_multichannel_rfdetr()
    spec = fusion_spec(detector.model.model.state_dict())
    if spec is None:
        raise ValueError("Missing feature-fusion specification")
    # Runtime model-builder patches do not propagate to independently spawned ranks.
    devices = kwargs.get("devices", 1)
    strategy = kwargs.get("strategy", "auto")
    if (
        devices not in (1, "1")
        or kwargs.get("num_nodes", 1) != 1
        or strategy not in ("auto", "single_device")
    ):
        raise ValueError(
            "Feature fusion currently supports single-device training only"
        )
    augmentation = kwargs.pop("augmentation", "default")
    if augmentation not in ("default", "none"):
        raise ValueError(
            "Feature fusion supports default/none torchvision augmentation"
        )
    if kwargs.get("aug_config") not in (None, {}):
        raise ValueError(
            "Nonempty Albumentations presets cannot preserve multichannel TIFFs"
        )
    if augmentation == "none":
        kwargs["aug_config"] = {}
    kwargs["augmentation_backend"] = "torchvision"
    kwargs.setdefault("devices", 1)
    kwargs.setdefault("save_dataset_grids", False)
    previous = detector.model_config.pretrain_weights
    # RF-DETR constructs a separate training module. Snapshot current weights,
    # including learned attention, so repeated training calls retain them.
    with tempfile.TemporaryDirectory(prefix="rfdetr-feature-fusion-") as folder:
        snapshot = Path(folder) / "initial.pth"
        save_adapted_model(detector, snapshot)
        detector.model_config.pretrain_weights = str(snapshot)
        try:
            with feature_fusion_training(spec), multichannel_datasets():
                RFDETR.train(detector, **kwargs)
        finally:
            detector.model_config.pretrain_weights = previous


def expand_rfdetr_feature_fusion(
    input_model: Path,
    output_model: Path,
    num_views: int,
    variant: str | None = None,
    reference_view: int = 0,
    hidden_dim: int = 64,
):
    import rfdetr

    if variant is None:
        detector = rfdetr.RFDETR.from_checkpoint(str(input_model), device="cpu")
    else:
        names = {
            "nano": "RFDETRNano",
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
            "base": "RFDETRBase",
            "large": "RFDETRLarge",
            "xlarge": "RFDETRXLarge",
            "2xlarge": "RFDETR2XLarge",
        }
        name = names.get(variant.lower(), variant)
        if not name.startswith("RFDETR") or not hasattr(rfdetr, name):
            raise ValueError(f"Unknown RF-DETR variant: {variant}")
        detector = getattr(rfdetr, name)(
            pretrain_weights=str(input_model), device="cpu"
        )
    adapt_model(detector, num_views, reference_view, hidden_dim)
    save_adapted_model(detector, output_model)
    return detector

def validate_saved_model(path: Path, expected_views: int) -> None:
    import torch

    detector = load_adapted_model(path)
    encoder = detector.model.model.backbone[0].encoder
    if encoder.num_views != expected_views:
        raise RuntimeError("Reloaded view count does not match")
    detector.model.model.eval()
    # Exercise the real encoder/attention/projector rather than only inspecting weights.
    block = detector.model_config.patch_size * detector.model_config.num_windows
    with torch.no_grad():
        features = encoder(torch.zeros(1, 3 * expected_views, block, block))
        projected = detector.model.model.backbone[0].projector(features)
    if not projected or any(not torch.isfinite(feature).all() for feature in projected):
        raise RuntimeError("Feature-fusion forward produced invalid features")
