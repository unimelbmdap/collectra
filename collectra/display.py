"""Artefact display protocol, independent of the desktop GUI.

``display(context)`` returns a JSON-compatible dict with ``kind`` set to
``image``, ``text``, or ``properties``. Image sources are PNG data URLs; image
annotations use viewer-relative normalized coordinates. Text formats are
``plain``, ``markdown``, or ``xml``. Custom classes may reuse these views.
"""

from pathlib import Path
from dataclasses import fields
import base64
import io

from collectra.utils import load_class_from_string


class DisplayContext:
    def __init__(self, records: dict, directory: Path):
        self.records = records
        self.directory = Path(directory).resolve()
        self.instances = {}

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.directory / path

    def artefact(self, node_id):
        from collectra.types.images import Image
        from collectra.types.links import Link
        from collectra.types.texts import Text

        if node_id in self.instances:
            return self.instances[node_id]
        record = dict(self.records[node_id])
        cls = load_class_from_string(record.pop("type"))
        record["name"] = record.pop("label", record.get("name", ""))
        if isinstance(record.get("parents"), str):
            record["parents"] = [record["parents"]]
        if issubclass(cls, Image):
            record["data"] = self.path(record.get("data", record.get("path", "")))
        elif issubclass(cls, Text):
            source = self.text_path(node_id)
            if source:
                record["data"] = source
        accepted = {f.name for f in fields(cls) if f.init}
        item = cls(**{k: v for k, v in record.items() if k in accepted})
        self.instances[node_id] = item
        if type(item) is Link:
            item.target = self.artefact(item.parent_id)
        return item

    def text_path(self, node_id):
        value = self.records[node_id].get("data", "")
        if not isinstance(value, (str, Path)) or not value:
            return None
        try:
            path = self.path(value)
            return path if path.is_file() else None
        except (OSError, ValueError):
            return None

    def text(self, item):
        path = self.text_path(item.id)
        format = {".md": "markdown", ".markdown": "markdown", ".xml": "xml"}.get(
            path.suffix.lower() if path else "", "plain"
        )
        return {
            "kind": "text",
            "format": format,
            "text": str(item.data),
            "editable": True,
            "target_id": item.id,
        }

    @staticmethod
    def publish_image(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def region(self, image, region, inverse=False):
        """Transform source-normalized boxes to/from the rotated crop viewer."""
        left, top, right, bottom = image.display_bounds()
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

    def annotations(self, image):
        from collectra.types.images import ImageCrop

        rows = []
        for node_id, record in self.records.items():
            parents = record.get("parents", [])
            parents = [parents] if isinstance(parents, str) else parents
            if image.id not in parents:
                continue
            try:
                if not issubclass(load_class_from_string(record["type"]), ImageCrop):
                    continue
            except (ImportError, AttributeError):
                continue
            child = self.artefact(node_id)
            if (
                not isinstance(child, ImageCrop)
                or child.get_path().resolve() != image.get_path().resolve()
            ):
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
                    "crop_region": self.region(image, region),
                }
            )
        return rows

    def image(self, item):
        from PIL import Image as PILImage
        from collectra.types.images import read_tiff_channels

        path = item.get_path()
        if path.suffix.lower() in {".tif", ".tiff"}:
            import numpy as np

            pixels = read_tiff_channels(path)
            # Default spectral preview: first RGB triplet, or first gray band.
            # Subclasses can override display() to choose a different projection.
            pixels = pixels[..., :3] if pixels.shape[2] >= 3 else pixels[..., 0]
            if pixels.dtype != np.uint8:
                pixels = pixels.astype(float)
                low, high = np.nanmin(pixels), np.nanmax(pixels)
                pixels = np.nan_to_num((pixels - low) / (high - low or 1) * 255).astype(
                    np.uint8
                )
            preview = PILImage.fromarray(pixels)
        else:
            with PILImage.open(path) as source:
                preview = source.convert("RGB")
        preview = preview.crop(item.display_bounds()).rotate(
            item.orientation.to_degree(), expand=True
        )
        return {
            "kind": "image",
            "source": self.publish_image(preview),
            "image_path": str(path),
            "target_id": item.id,
            "annotations": self.annotations(item),
            "can_create_crop": True,
        }
