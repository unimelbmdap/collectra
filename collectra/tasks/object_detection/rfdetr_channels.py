"""Load and train RF-DETR checkpoints whose input layer has more than three channels.

RF-DETR loads checkpoint weights into a 3-channel model and only afterwards
widens the input layer to ``num_channels``, so a checkpoint whose input layer is
already wider (e.g. one written by ``adapt-rfdetr.py`` or by multi-channel
training) cannot be loaded through ``pretrain_weights``. Its Lightning training
module also never widens the input layer at all.

``multichannel_checkpoints()`` wraps ``load_pretrain_weights`` so the model's
patch projection is resized to the checkpoint's channel count before loading,
and, for training, widened to ``model_config.num_channels`` afterwards using
RF-DETR's own filter repetition.
"""

from __future__ import annotations

import contextlib
import copy
import os
from typing import Iterator

import torch

from .rfdetr_data import channel_stats

PROJECTION_KEY = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"


def _patch_embeddings(nn_model):
    return nn_model.backbone[0].encoder.encoder.embeddings.patch_embeddings


def checkpoint_channels(path) -> int:
    """Input channels of a checkpoint's patch projection (3 if it cannot be read)."""
    from rfdetr.utilities.io import _safe_torch_load

    if path is None or not os.path.isfile(str(path)):
        return 3
    checkpoint = _safe_torch_load(str(path))
    state = checkpoint.get("model")
    if state is None:
        state = {
            key.removeprefix("model.").removeprefix("_orig_mod."): value
            for key, value in checkpoint.get("state_dict", {}).items()
        }
    weight = state.get(PROJECTION_KEY)
    return 3 if weight is None else int(weight.shape[1])


def set_input_channels(nn_model, channels: int) -> None:
    """Widen or narrow the patch projection, repeating pretrained filters as RF-DETR does."""
    from rfdetr.inference import _adapt_input_conv

    embeddings = _patch_embeddings(nn_model)
    projection = embeddings.projection
    if projection.in_channels == channels:
        return
    if projection.in_channels != 3:
        raise ValueError(
            f"Cannot adapt a {projection.in_channels}-channel input layer to {channels} channels"
        )
    new_projection = copy.deepcopy(projection)
    new_projection.in_channels = channels
    new_projection.weight = torch.nn.Parameter(
        _adapt_input_conv(channels, projection.weight),
        requires_grad=projection.weight.requires_grad,
    )
    embeddings.projection = new_projection
    embeddings.num_channels = channels


def sync_channels(detector) -> int:
    """Align an RFDETR wrapper's config and normalization with its input layer.

    Wrappers without a DINOv2 patch projection (e.g. test doubles) are left unchanged.
    """
    try:
        channels = _patch_embeddings(detector.model.model).projection.in_channels
    except (AttributeError, IndexError, TypeError):
        return getattr(getattr(detector, "model_config", None), "num_channels", 3)
    detector.model_config.num_channels = channels
    args = getattr(detector.model, "args", None)
    if args is not None and hasattr(args, "num_channels"):
        args.num_channels = channels
    mean, std = channel_stats(channels)
    detector.means, detector.stds = list(mean), list(std)
    return channels


def _channel_aware(load_pretrain_weights, widen_after: bool):
    def load(nn_model, model_config, *args, **kwargs):
        stored = checkpoint_channels(model_config.pretrain_weights)
        if stored != 3:
            set_input_channels(nn_model, stored)
        result = load_pretrain_weights(nn_model, model_config, *args, **kwargs)
        if widen_after and model_config.num_channels != 3:
            set_input_channels(nn_model, model_config.num_channels)
        return result

    return load


@contextlib.contextmanager
def multichannel_checkpoints() -> Iterator[None]:
    """Let RF-DETR load, and train from, checkpoints with any number of input channels."""
    import rfdetr.inference as inference

    # The inference builder widens the input layer itself after loading.
    patches = [(inference, False)]
    try:
        import rfdetr.training.module_model as module_model

        patches.append((module_model, True))
    except ImportError:
        pass

    originals = [(module, module.load_pretrain_weights) for module, _ in patches]
    for module, widen_after in patches:
        module.load_pretrain_weights = _channel_aware(
            module.load_pretrain_weights, widen_after
        )
    try:
        yield
    finally:
        for module, original in originals:
            module.load_pretrain_weights = original
