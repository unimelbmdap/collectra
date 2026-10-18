"""Shared RGB classifier encoding with attention over stacked RGB views."""

import torch
from torch import nn


def _take_projection(model):
    """Separate the final linear projection without changing its input features."""
    for name in ("fc", "classifier", "heads", "head"):
        head = getattr(model, name, None)
        if isinstance(head, nn.Linear):
            setattr(model, name, nn.Identity())
            return head
        if isinstance(head, nn.Sequential):
            projections = [
                (path, layer)
                for path, layer in head.named_modules()
                if isinstance(layer, (nn.Linear, nn.Conv2d))
            ]
            if not projections or not isinstance(projections[-1][1], nn.Linear):
                continue
            path, projection = projections[-1]
            parent_path, _, child = path.rpartition(".")
            parent = head.get_submodule(parent_path) if parent_path else head
            setattr(parent, child, nn.Identity())
            return projection
    raise ValueError(
        "Classifier fusion requires a final linear classification head (for example ResNet or ConvNeXt)"
    )


class RGBViewFusionClassifier(nn.Module):
    """Encode B*V RGB images with one encoder, attend across V, classify once."""

    def __init__(self, model, input_channels, *, reference_view=0, hidden_dim=128):
        super().__init__()
        if input_channels < 3 or input_channels % 3:
            raise ValueError(
                "fusion requires stacked RGB views: input channels must be a positive multiple of 3"
            )
        views = input_channels // 3
        if not 0 <= reference_view < views or hidden_dim < 1:
            raise ValueError(
                "Invalid classifier fusion reference view or hidden dimension"
            )
        first = next(
            (layer for layer in model.modules() if isinstance(layer, nn.Conv2d)), None
        )
        if first is not None and first.in_channels != 3:
            raise ValueError(
                "fusion requires an RGB backbone, not an expanded first convolution; start from an RGB model or a fusion checkpoint"
            )
        self.head = _take_projection(model)
        self.encoder = model
        self.input_channels = input_channels
        self.num_views = views
        self.reference_view = reference_view
        self.hidden_dim = hidden_dim
        features = self.head.in_features
        self.attention = nn.Sequential(
            nn.Linear(2 * features, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)
        self.view_bias = nn.Parameter(torch.zeros(views))
        self.gate = nn.Parameter(torch.zeros(()))
        self.to(device=self.head.weight.device, dtype=self.head.weight.dtype)
        self.train(model.training)

    @property
    def fusion_spec(self):
        return {
            "version": 1,
            "num_views": self.num_views,
            "reference_view": self.reference_view,
            "hidden_dim": self.hidden_dim,
        }

    def forward(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1] != self.input_channels:
            raise ValueError(
                f"Fusion classifier expects BCHW with {self.input_channels} channels; got {tuple(pixels.shape)}"
            )
        batch, _, height, width = pixels.shape
        encoded = self.encoder(pixels.reshape(batch * self.num_views, 3, height, width))
        if encoded.ndim != 2 or encoded.shape[1] != self.head.in_features:
            raise ValueError(
                "Fusion classifier encoder must return one feature vector per RGB view"
            )
        features = encoded.reshape(batch, self.num_views, -1)
        reference = features[:, self.reference_view]
        query = reference[:, None].expand_as(features)
        logits = self.attention(torch.cat([features, query], dim=-1)).squeeze(-1)
        weights = (logits + self.view_bias).float().softmax(dim=1).to(features.dtype)
        pooled = (features * weights[..., None]).sum(dim=1)
        fused = reference + self.gate.tanh() * (pooled - reference)
        return self.head(fused)


def restore_fusion(model, input_channels, spec):
    if not isinstance(spec, dict) or spec.get("version") != 1:
        raise ValueError("Unsupported torchvision classifier fusion checkpoint version")
    if spec.get("num_views", 0) * 3 != input_channels:
        raise ValueError("Fusion checkpoint view count does not match input_channels")
    return RGBViewFusionClassifier(
        model,
        input_channels,
        reference_view=spec["reference_view"],
        hidden_dim=spec["hidden_dim"],
    )
