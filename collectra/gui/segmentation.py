"""Validate and atomically persist circular mask brush strokes."""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def paint_mask(item, strokes, orientation):
    from collectra.types.images import ImageSegmentation

    if not isinstance(item, ImageSegmentation):
        raise ValueError("Only ImageSegmentation masks can be painted")
    if orientation != item.orientation.to_string():
        raise ValueError(
            "The image orientation changed; reopen the segmentation before painting"
        )
    if not isinstance(strokes, list) or not strokes:
        raise ValueError("No mask strokes supplied")
    pixels = item._mask_image().point(lambda value: 1 if value else 0)
    pixels = pixels.rotate(item.orientation.to_degree(), expand=True)
    before = np.asarray(pixels).copy()
    draw = ImageDraw.Draw(pixels)
    for stroke in strokes:
        if stroke.get("tool") not in {"brush", "eraser"}:
            raise ValueError("Unknown mask tool")
        radius = stroke.get("radius")
        if (
            type(radius) not in (float, int)
            or not math.isfinite(radius)
            or not 0.5 <= radius <= 4096
        ):
            raise ValueError("Invalid brush radius")
        points = stroke.get("points")
        if not isinstance(points, list) or not points or len(points) > 100000:
            raise ValueError("Invalid brush points")
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in point
                )
            ):
                raise ValueError("Invalid brush point")
            if (
                not -radius <= point[0] <= pixels.width + radius
                or not -radius <= point[1] <= pixels.height + radius
            ):
                raise ValueError("Brush point is outside the source image")
        fill = 1 if stroke["tool"] == "brush" else 0
        if len(points) > 1:
            draw.line(
                [tuple(p) for p in points], fill=fill, width=max(1, round(radius * 2))
            )
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    if pixels.getbbox() is None:
        raise ValueError(
            "A segmentation mask cannot be empty; leave at least one pixel"
        )
    if np.array_equal(before, np.asarray(pixels)):
        return False
    pixels = pixels.rotate(-item.orientation.to_degree(), expand=True)
    path = Path(item._mask_path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".png", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pixels.save(stream, format="PNG")
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True
