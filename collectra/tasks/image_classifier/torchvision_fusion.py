"""Shared RGB classifier encoding with spatial attention over stacked RGB views.

Every view passes through one shared RGB encoder. Immediately before global
pooling, softmax attention chooses how much each view contributes at each
spatial position. No view is privileged: attention is conditioned on the mean
of all views, so the prediction does not depend on the order of the views.
"""

import torch
from torch import nn

FORMAT_VERSION = 2
PROBE_SIZE = 224


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


def _pooled_channels(model, pool):
    """Run a dummy RGB image to find the channel count of the map entering avgpool."""
    shapes = []
    handle = pool.register_forward_pre_hook(
        lambda module, args: shapes.append(args[0].shape)
    )
    parameter = next(model.parameters())
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            model(
                torch.zeros(
                    1,
                    3,
                    PROBE_SIZE,
                    PROBE_SIZE,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    finally:
        handle.remove()
        for module, training in modes:
            module.training = training
    if len(shapes) != 1 or len(shapes[0]) != 4:
        raise ValueError(
            "Classifier fusion requires an avgpool layer that pools a spatial feature map"
        )
    return shapes[0][1]


class SpatialViewAttention(nn.Module):
    """Per-position softmax attention over views, conditioned on the view mean."""

    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Begin with uniform weights: the fused map is the mean of the views.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def weights(self, features):
        """Return B,V,H,W weights that sum to one over views at each position."""
        batch, views, channels, height, width = features.shape
        context = features.mean(dim=1, keepdim=True).expand_as(features)
        paired = torch.cat((features, context), dim=2).reshape(
            batch * views, 2 * channels, height, width
        )
        logits = self.score(paired).reshape(batch, views, height, width)
        return logits.float().softmax(dim=1).to(features.dtype)

    def forward(self, features):
        return (features * self.weights(features)[:, :, None]).sum(dim=1)


class ViewFusionPool(nn.Module):
    """Fuse B*V feature maps into B maps, then apply the encoder's original pool."""

    def __init__(self, attention, pool, num_views: int):
        super().__init__()
        self.attention = attention
        self.pool = pool
        self.num_views = num_views

    def forward(self, features):
        views = features.reshape(-1, self.num_views, *features.shape[1:])
        return self.pool(self.attention(views))


class RGBViewFusionClassifier(nn.Module):
    """Encode B*V RGB images with one encoder, fuse spatially across V, classify once."""

    def __init__(self, model, input_channels, *, hidden_dim=128):
        super().__init__()
        if input_channels < 3 or input_channels % 3:
            raise ValueError(
                "fusion requires stacked RGB views: input channels must be a positive multiple of 3"
            )
        if hidden_dim < 1:
            raise ValueError("Invalid classifier fusion hidden dimension")
        first = next(
            (layer for layer in model.modules() if isinstance(layer, nn.Conv2d)), None
        )
        if first is not None and first.in_channels != 3:
            raise ValueError(
                "fusion requires an RGB backbone, not an expanded first convolution; start from an RGB model or a fusion checkpoint"
            )
        pool = getattr(model, "avgpool", None)
        if not isinstance(pool, nn.Module):
            raise ValueError(
                "Classifier fusion requires a convolutional model with an avgpool layer (for example ResNet, ConvNeXt or EfficientNet)"
            )
        channels = _pooled_channels(model, pool)
        self.head = _take_projection(model)
        self.num_views = input_channels // 3
        model.avgpool = ViewFusionPool(
            SpatialViewAttention(channels, hidden_dim), pool, self.num_views
        )
        self.encoder = model
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.to(device=self.head.weight.device, dtype=self.head.weight.dtype)
        # Match the encoder's mode without overriding its submodules' own modes.
        self.training = model.training
        model.avgpool.train(model.training)

    @property
    def attention(self):
        return self.encoder.avgpool.attention

    @property
    def fusion_spec(self):
        return {
            "version": FORMAT_VERSION,
            "num_views": self.num_views,
            "hidden_dim": self.hidden_dim,
        }

    def forward(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1] != self.input_channels:
            raise ValueError(
                f"Fusion classifier expects BCHW with {self.input_channels} channels; got {tuple(pixels.shape)}"
            )
        batch, _, height, width = pixels.shape
        rgb = pixels.reshape(batch * self.num_views, 3, height, width)
        return self.head(self.encoder(rgb))


def restore_fusion(model, input_channels, spec):
    if not isinstance(spec, dict) or spec.get("version") != FORMAT_VERSION:
        raise ValueError(
            "Unsupported torchvision classifier fusion checkpoint version; retrain with the current fusion"
        )
    if spec.get("num_views", 0) * 3 != input_channels:
        raise ValueError("Fusion checkpoint view count does not match input_channels")
    return RGBViewFusionClassifier(model, input_channels, hidden_dim=spec["hidden_dim"])
