"""GUI mask descriptors; extraction continues to use transparent cropped images."""

from PIL import Image
from .preview import bounded


def display_bounds(item):
    """Keep the full source available for expanding a segmentation mask."""
    from collectra.types.images import ImageSegmentation

    if isinstance(item, ImageSegmentation):
        return (0, 0, item.raw_width, item.raw_height)
    return item.display_bounds()


def segmentation(context, item):
    alpha = item._mask_image().rotate(item.orientation.to_degree(), expand=True)
    left, top, right, bottom = alpha.getbbox()
    width, height = alpha.size
    alpha = bounded(alpha, mask=True, max_pixels=context.max_pixels)
    mask = Image.new("RGBA", alpha.size, "white")
    mask.putalpha(alpha)
    return {
        "mask_source": context.publish_image(mask),
        "width": width,
        "height": height,
        "preview_width": alpha.width,
        "preview_height": alpha.height,
        "orientation": item.orientation.to_string(),
        "bounds": {"x": left, "y": top, "w": right - left, "h": bottom - top},
    }
