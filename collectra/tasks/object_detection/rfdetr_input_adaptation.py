"""Adapt RGB RF-DETR checkpoints to concatenated aligned views."""

from pathlib import Path

import torch


def _patch_embeddings(detector):
    try:
        embeddings = detector.model.model.backbone[
            0
        ].encoder.encoder.embeddings.patch_embeddings
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Could not locate RF-DETR's DINOv2 patch embeddings."
        ) from exc
    if not isinstance(embeddings.projection, torch.nn.Conv2d):
        raise RuntimeError("Expected the patch projection to be torch.nn.Conv2d.")
    return embeddings


def _expand_model(detector, num_views: int) -> None:
    if num_views < 1:
        raise ValueError("num_views must be >= 1")
    embeddings = _patch_embeddings(detector)
    old_conv = embeddings.projection
    if old_conv.in_channels != 3:
        raise ValueError(
            "Expected a 3-channel pretrained model, but the patch projection "
            f"has {old_conv.in_channels} input channels."
        )
    if old_conv.groups != 1:
        raise RuntimeError("Grouped input convolutions are not supported.")

    channels = 3 * num_views
    new_conv = torch.nn.Conv2d(
        channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.repeat(1, num_views, 1, 1) / num_views)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    new_conv.weight.requires_grad_(old_conv.weight.requires_grad)
    if old_conv.bias is not None:
        new_conv.bias.requires_grad_(old_conv.bias.requires_grad)
    new_conv.train(old_conv.training)
    embeddings.projection = new_conv
    # DINOv2 checks this attribute before calling the convolution.
    embeddings.num_channels = channels
    detector.model.model.backbone[0].encoder.encoder.config.num_channels = channels
    detector.model_config.num_channels = channels
    detector.model.args.num_channels = channels
    detector.means = list(detector.means[:3]) * num_views
    detector.stds = list(detector.stds[:3]) * num_views


def expand_rfdetr_input_channels(
    input_model: Path,
    output_model: Path,
    num_views: int,
    variant: str | None = None,
) -> None:
    """Save expanded model weights and reconstruction metadata (no optimizer)."""
    if num_views < 1:
        raise ValueError("num_views must be >= 1")
    import rfdetr

    if variant is None:
        try:
            detector = rfdetr.RFDETR.from_checkpoint(str(input_model), device="cpu")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Could not reconstruct the input checkpoint automatically. "
                "Use --variant with its model size, e.g. --variant nano. "
                f"RF-DETR reported: {exc}"
            ) from exc
    else:
        variants = {
            "nano": "RFDETRNano",
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
            "base": "RFDETRBase",
            "large": "RFDETRLarge",
            "xlarge": "RFDETRXLarge",
            "2xlarge": "RFDETR2XLarge",
        }
        name = variants.get(variant.lower(), variant)
        model_class = getattr(rfdetr, name, None)
        if model_class is None or not name.startswith("RFDETR"):
            raise ValueError(f"Unknown or unavailable RF-DETR variant: {variant}")
        detector = model_class(pretrain_weights=str(input_model), device="cpu")

    _expand_model(detector, num_views)
    config = detector.model_config.model_dump(mode="json")
    args = dict(config)
    args["class_names"] = list(detector.model.class_names or [])
    checkpoint = {
        "model": detector.model.model.state_dict(),
        "model_name": type(detector).__name__,
        "model_config": config,
        "args": args,
        "multichannel": {
            "num_views": num_views,
            "num_channels": 3 * num_views,
            "means": detector.means,
            "stds": detector.stds,
        },
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_model)


def load_adapted_model(output_model: Path, device: str = "cpu"):
    """Reconstruct a checkpoint produced here, expanding before loading weights.

    Returns the RF-DETR wrapper, including repeated RGB normalization statistics.
    Use CHW tensors with 3 * num_views channels for prediction.
    """
    import rfdetr

    checkpoint = torch.load(output_model, map_location="cpu", weights_only=True)
    metadata = checkpoint["multichannel"]
    num_views = metadata["num_views"]
    if num_views < 1 or metadata["num_channels"] != 3 * num_views:
        raise ValueError("Inconsistent multichannel checkpoint metadata.")
    config = dict(checkpoint["model_config"])
    # Avoid the upstream order: loading expanded weights into an RGB layer.
    config.update(pretrain_weights=None, num_channels=3, device=device)
    model_class = getattr(rfdetr, checkpoint["model_name"])
    detector = model_class(**config)
    _expand_model(detector, num_views)
    detector.model.model.load_state_dict(checkpoint["model"], strict=True)
    detector.model.class_names = checkpoint["args"].get("class_names", [])
    detector.means = list(metadata["means"])
    detector.stds = list(metadata["stds"])
    return detector


def validate_saved_model(output_model: Path, expected_channels: int) -> None:
    reloaded = load_adapted_model(output_model)
    embeddings = _patch_embeddings(reloaded)
    conv = embeddings.projection
    if conv.in_channels != expected_channels:
        raise RuntimeError(
            f"Reloaded model has {conv.in_channels} input channels; "
            f"expected {expected_channels}."
        )
    # Exercise the channel guard as well as the convolution after strict reload.
    with torch.no_grad():
        embeddings(torch.zeros(1, expected_channels, *conv.kernel_size))
