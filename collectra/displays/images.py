"""Image previews, crop coordinates and child image annotations."""

from collections import OrderedDict
from collectra.utils import load_class_from_string
from .image_segmentation import display_bounds
from .preview import bounded, open_preview

# Share the few most recent encodings across contexts and panels.
_PREVIEW_CACHE_SIZE = 6
_preview_cache = OrderedDict()


def region(context, image, region, inverse=False):
    """Transform source-normalized boxes to/from the rotated crop viewer."""
    left, top, right, bottom = display_bounds(image)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("Cannot display an empty image crop")
    x, y = region["x_center"], region["y_center"]
    w, h = region["width_relative"], region["height_relative"]
    turns = (image.orientation.to_degree() // 90) % 4
    if inverse:
        for _ in range((-turns) % 4):
            x, y, w, h = y, 1 - x, h, w
        x, y = (left + x * width) / image.raw_width, (
            top + y * height
        ) / image.raw_height
        w, h = w * width / image.raw_width, h * height / image.raw_height
    else:
        x, y = (x * image.raw_width - left) / width, (
            y * image.raw_height - top
        ) / height
        w, h = w * image.raw_width / width, h * image.raw_height / height
        for _ in range(turns):
            x, y, w, h = y, 1 - x, h, w
    return dict(x_center=x, y_center=y, width_relative=w, height_relative=h)


def annotations(context, image):
    from collectra.types.images import ImageCrop, ImageSegmentation

    rows = []
    for node_id, record in context.records.items():
        parents = record.get("parents", [])
        parents = [parents] if isinstance(parents, str) else parents
        if image.id not in parents:
            continue
        try:
            if not issubclass(
                load_class_from_string(record["type"]),
                (ImageCrop, ImageSegmentation),
            ):
                continue
        except (ImportError, AttributeError):
            continue
        child = context.artefact(node_id)
        if (
            not isinstance(child, (ImageCrop, ImageSegmentation))
            or child.get_path().resolve() != image.get_path().resolve()
        ):
            continue
        if isinstance(child, ImageSegmentation):
            from PIL import Image as PILImage, ImageChops

            alpha = child._mask_image()
            if isinstance(image, ImageSegmentation):
                alpha = ImageChops.multiply(alpha, image._mask_image())
            alpha = alpha.crop(display_bounds(image)).rotate(
                image.orientation.to_degree(), expand=True
            )
            bounds = alpha.getbbox()
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            cropped = bounded(alpha.crop(bounds), mask=True, max_pixels=context.max_pixels)
            overlay = PILImage.new("RGBA", cropped.size, "white")
            overlay.putalpha(cropped)
            rows.append(
                {
                    "id": node_id,
                    "type": record["type"],
                    "label": record.get("label", ""),
                    "name": record.get("name", ""),
                    "mask_source": context.publish_image(overlay),
                    "mask_region": {
                        "x": left / alpha.width,
                        "y": top / alpha.height,
                        "width": (right - left) / alpha.width,
                        "height": (bottom - top) / alpha.height,
                    },
                }
            )
            continue
        region = {
            key: getattr(child, key)
            for key in ("x_center", "y_center", "width_relative", "height_relative")
        }
        rows.append(
            {
                "id": node_id,
                "type": record["type"],
                "label": record.get("label", ""),
                "name": record.get("name", ""),
                "confidence": child.confidence,
                "crop_region": context.region(image, region),
            }
        )
    return rows


def image(context, item):
    from PIL import Image as PILImage

    from collectra.types.images import (
        ImageSegmentation,
        image_channel_count,
        read_tiff_channels,
    )

    path = item.get_path()
    view_count = 1
    key = _preview_key(context, item, path)
    if key in _preview_cache:
        _preview_cache.move_to_end(key)
        source, view_count = _preview_cache[key]
        return _image_view(context, item, path, source, view_count)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import numpy as np
        from tifffile import TiffFile

        channels = image_channel_count(path)
        # Only complete RGB triplets form multiple views. Other spectral
        # layouts keep the first-triplet/gray preview and can override display.
        if channels >= 3 and channels % 3 == 0:
            view_count = channels // 3
        if context.rgb_view >= view_count:
            raise ValueError(f"RGB view index must be less than {view_count}")
        start = context.rgb_view * 3
        stop = start + (3 if channels >= 3 else 1)
        with TiffFile(path) as tiff:
            series = tiff.series[0]
            # CYX/QYX stacks store each channel on a separate page. Decode
            # just this view instead of all 75 channels on every button click.
            pages = series.pages
            if len(pages) == channels and all(
                pages[i].axes == "YX" for i in range(start, stop)
            ):
                pixels = np.stack(
                    [pages[i].asarray() for i in range(start, stop)], axis=-1
                )
            else:
                pixels = read_tiff_channels(path)[..., start:stop]
        if pixels.shape[2] == 1:
            pixels = pixels[..., 0]
        if pixels.dtype != np.uint8:
            pixels = pixels.astype(float)
            low, high = np.nanmin(pixels), np.nanmax(pixels)
            pixels = np.nan_to_num((pixels - low) / (high - low or 1) * 255).astype(
                np.uint8
            )
        preview = PILImage.fromarray(pixels)
    else:
        if context.rgb_view:
            raise ValueError("This image has only one RGB view")
        preview = open_preview(path, display_bounds(item), "RGBA" if isinstance(item, ImageSegmentation) else "RGB", max_pixels=context.max_pixels)
    if path.suffix.lower() in {".tif", ".tiff"}:
        preview = bounded(preview.crop(display_bounds(item)), max_pixels=context.max_pixels)
    preview = preview.rotate(
        item.orientation.to_degree(), expand=True
    )
    source = context.publish_image(preview)
    if key is not None:
        _preview_cache[key] = (source, view_count)
        while len(_preview_cache) > _PREVIEW_CACHE_SIZE:
            _preview_cache.popitem(last=False)
    return _image_view(context, item, path, source, view_count)


def _preview_key(context, item, path):
    """What the encoded preview depends on, or None when it is not cached
    (segmentations also depend on their mask file)."""
    from collectra.types.images import ImageSegmentation

    if isinstance(item, ImageSegmentation):
        return None
    try:
        stat = path.stat()
        return (
            str(path.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            context.rgb_view,
            context.max_pixels,
            tuple(item.display_bounds()),
            item.orientation.to_degree(),
        )
    except OSError:
        return None


def _image_view(context, item, path, source, view_count):
    from collectra.types.images import ImageSegmentation
    from .image_segmentation import segmentation

    view = {
        "kind": "image",
        "source": source,
        "image_path": str(path),
        "target_id": item.id,
        "annotations": context.annotations(item),
        "can_create_crop": True,
        "can_rotate": True,
        "rgb_view_index": context.rgb_view,
        "rgb_view_count": view_count,
    }
    if isinstance(item, ImageSegmentation):
        view["segmentation"] = segmentation(context, item)
    return view
