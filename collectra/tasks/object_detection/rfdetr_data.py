"""Multi-channel TIFF support for the RF-DETR training data pipeline.

RF-DETR's ``CocoDetection`` decodes every image with ``PIL.Image.convert("RGB")``,
which drops channels beyond three and cannot open most multispectral TIFFs. This
module swaps the built datasets for a subclass that decodes TIFFs with
``read_tiff_channels`` into CHW tensors, and replaces the ImageNet ``Normalize``
with one that matches the image's channel count.

Kept separate from ``rfdetr.py`` so that importing the task does not import
``rfdetr``, and so DataLoader worker processes can unpickle these classes.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from torchvision import tv_tensors

from collectra.types.images import read_tiff_channels

TIFF_SUFFIXES = {".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def channel_stats(channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """ImageNet statistics for the first three channels; their average for any others."""
    extra_mean = sum(IMAGENET_MEAN) / 3
    extra_std = sum(IMAGENET_STD) / 3
    mean = IMAGENET_MEAN[:channels] + (extra_mean,) * max(0, channels - 3)
    std = IMAGENET_STD[:channels] + (extra_std,) * max(0, channels - 3)
    return mean, std


class ChannelNormalize(Normalize):
    """RF-DETR ``Normalize`` whose mean/std follow the image's channel count."""

    def __call__(self, image, target=None):
        channels = image.shape[-3]
        if channels != len(IMAGENET_MEAN):
            mean, std = channel_stats(channels)
            normalize = Normalize(mean, std)
            return normalize(image, target)
        return super().__call__(image, target)


class _ImageSize:
    """Stand-in exposing only ``.size``, the one attribute ``ConvertCoco`` reads."""

    def __init__(self, width: int, height: int):
        self.size = (width, height)


class MultiChannelCocoDetection(CocoDetection):
    """``CocoDetection`` that keeps every channel of TIFF images."""

    def _image_path(self, image_id: int) -> Path:
        return Path(self.root) / self.coco.loadImgs(image_id)[0]["file_name"]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        image_id = self.ids[idx]
        path = self._image_path(image_id)
        if path.suffix.lower() not in TIFF_SUFFIXES:
            return super().__getitem__(idx)

        pixels = read_tiff_channels(path)
        # torch lacks CPU kernels (e.g. flip) for uint16/uint32, so scale integer
        # types other than uint8 to [0, 1] floats here, as ToDtype(scale=True) would.
        if np.issubdtype(pixels.dtype, np.integer) and pixels.dtype != np.uint8:
            pixels = pixels.astype(np.float32) / np.iinfo(pixels.dtype).max
        elif pixels.dtype != np.uint8:
            pixels = pixels.astype(np.float32)
        img = tv_tensors.Image(torch.from_numpy(pixels.transpose(2, 0, 1).copy()))
        height, width = pixels.shape[:2]
        target = {"image_id": image_id, "annotations": self._load_target(image_id)}
        _, target = self.prepare(_ImageSize(width, height), target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def _adapt_dataset(dataset):
    if type(dataset) is not CocoDetection:
        return dataset
    dataset.__class__ = MultiChannelCocoDetection
    transforms = getattr(dataset._transforms, "transforms", [])
    has_tiffs = any(
        Path(image["file_name"]).suffix.lower() in TIFF_SUFFIXES
        for image in dataset.coco.imgs.values()
    )
    if has_tiffs and any(isinstance(t, AlbumentationsWrapper) for t in transforms):
        # The Albumentations wrapper round-trips images through PIL, which
        # cannot hold arbitrary channel counts.
        raise ValueError(
            "TIFF inputs are only supported with the default (torchvision) "
            "augmentation pipeline; use augmentation='default'."
        )
    for index, transform in enumerate(transforms):
        if type(transform) is Normalize:
            transforms[index] = ChannelNormalize()
    return dataset


@contextlib.contextmanager
def multichannel_datasets() -> Iterator[None]:
    """Make RF-DETR training build datasets that keep all TIFF channels."""
    import rfdetr.training.module_data as module_data

    original = module_data.build_dataset

    def build_dataset(image_set: str, args: Any, resolution: int):
        return _adapt_dataset(original(image_set, args, resolution))

    module_data.build_dataset = build_dataset
    try:
        yield
    finally:
        module_data.build_dataset = original
