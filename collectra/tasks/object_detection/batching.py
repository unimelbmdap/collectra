"""Helpers shared by detectors that predict over many images in one batch."""

from __future__ import annotations

from collectra.types.images import Image, ImageCrop


def image_cache_key(image: Image, *settings) -> tuple:
    """Identify the pixels (and settings) a cached prediction was made for."""
    parent = getattr(image, "source_parent", None)
    crop = (
        (image.x_center, image.y_center, image.width_relative, image.height_relative)
        if isinstance(image, ImageCrop)
        else None
    )
    return (
        str(image.get_path()),
        crop,
        image.orientation,
        image_cache_key(parent) if parent is not None else None,
        *settings,
    )
