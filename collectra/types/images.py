"""Image handling and processing classes for the Collectra system.

This module provides the core image classes used throughout Collectra for
handling image data, metadata, and cropping operations. It supports various
image formats and provides utilities for encoding, metadata extraction,
and coordinate-based cropping.

The module includes:
    - Base Image class for general image handling
    - ImageCrop class for coordinate-based image cropping
    - Image format validation and metadata extraction
    - Base64 encoding for image serialization

Classes:
    Image: Base class for image handling and metadata
    ImageCrop: Specialized class for cropped image regions
"""

__all__ = ["Image", "ImageCrop"]


import base64
import io
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image as ImagePil

from .base import Artefact


class Orientation(Enum):
    """Enumeration for image orientation states."""

    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3

    def to_string(self) -> str:
        match self.value:
            case 0:
                return "north"
            case 1:
                return "west"
            case 2:
                return "south"
            case 3:
                return "east"

    def to_degree(self) -> int:
        match self.value:
            case 0:
                return 0
            case 1:
                return -90
            case 2:
                return -180
            case 3:
                return -270

    @staticmethod
    def from_string(direction: str) -> "Orientation":
        direction = direction.strip().lower()
        match direction:
            case "north":
                return Orientation.NORTH
            case "west":
                return Orientation.WEST
            case "south":
                return Orientation.SOUTH
            case "east":
                return Orientation.EAST
            case _:
                raise ValueError(f"Invalid degree for orientation: {direction}")


@dataclass
class Image(Artefact):
    """Base class for handling image data and metadata in Collectra workflows.

    Provides core functionality for loading, encoding, and managing image files
    with support for various formats and metadata extraction. Serves as the
    foundation for more specialized image processing classes.

    Attributes:
        path (str): File system path to the image file.
        raw_width (int): Width of the image in pixels.
        raw_height (int): Height of the image in pixels.
        format (str | None): Image format (PNG, JPEG, etc.) or None if unknown.
    """

    data: str | Path | ImagePil.Image = field(default="")  # Path to the image file
    raw_width: int = field(init=False, default=0)  # Image width in pixels
    raw_height: int = field(init=False, default=0)  # Image height in pixels
    ext: str | None = field(default="")  # Image format (e.g., PNG, JPEG)
    embeddings: str | list[str] = field(default_factory=list)  # Optional embedding data
    orientation: Orientation = field(default=Orientation.NORTH)  # Image orientation

    def attributes_to_ignore(self):
        attributes = super().attributes_to_ignore()
        [attributes.add(attr) for attr in ["raw_width", "raw_height", "ext"]]
        return attributes

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["orientation"] = (
            self.orientation.to_string()
            if isinstance(self.orientation, Orientation)
            else self.orientation
        )
        return serialized

    def save(self, path: Path | str = None):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Simply copy to new dest
        shutil.copy(self.get_path(), path)

    @staticmethod
    def image_types() -> list[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]

    @property
    def width(self) -> int | float:
        return self.raw_width

    @property
    def height(self) -> int | float:
        return self.raw_height

    def __post_init__(self):
        """Create an Image instance from a file path.

        Loads image metadata (dimensions and format) from the specified file
        and returns a configured Image instance.

        Args:
            data (Path): Path to the image file to load.

        Returns:
            Image: Configured Image instance with loaded metadata.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            PIL.UnidentifiedImageError: If the file is not a valid image.
        """
        super().__post_init__()
        if not self.data:
            raise ValueError("Image data path is empty.")
        if not isinstance(self.data, (str, Path)):
            raise TypeError("Image data must be a file path.")
        if not isinstance(self.data, Path):
            self.data = Path.cwd() / self.data
        if not self.data.exists() or not self.data.is_file():
            raise FileNotFoundError(f"Image file not found: {self.data}")
        if not isinstance(self.embeddings, list):
            self.embeddings = [self.embeddings]
        if isinstance(self.orientation, str):
            self.orientation = Orientation.from_string(self.orientation)
        with ImagePil.open(self.data) as imf:
            self.raw_width, self.raw_height = imf.size
            self.ext = imf.format

    def __call__(self) -> str:
        return str(self.get_path())

    def get_path(self) -> Path:
        """Get the file path of the image.

        Returns:
            Path: The file path of the image.
        """
        if not isinstance(self.data, Path):
            if not isinstance(self.data, Image):
                raise Exception(
                    "This appear to be not an Image object. Artefact path is only available for Image object."
                )
            raise Exception("Image data is not a valid Path object.")
        return self.data

    @staticmethod
    def is_image_file(path: Path) -> bool:
        """Check if a file path represents a supported image format.

        Args:
            path (Path): File path to check.

        Returns:
            bool: True if the file extension indicates a supported image format.
        """
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
        return path.suffix.lower() in image_extensions

    def _load_buffer(self, img: ImagePil.Image) -> bytes:
        buffer = io.BytesIO()
        img.save(buffer, format=self.ext)
        buffer.seek(0)
        return buffer.read()

    def load(self) -> bytes:
        """Load the image file content as raw bytes.

        Returns:
            bytes: Raw binary content of the image file.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            PermissionError: If the file cannot be read due to permissions.
        """
        img = self.pil()
        return self._load_buffer(img)

    def get_encoding(self) -> str:
        """Get the base64 encoded representation of the image.

        Loads the image content and encodes it as a base64 string,
        suitable for embedding in JSON or other text-based formats.

        Returns:
            str: Base64 encoded string representation of the image.
        """
        return base64.b64encode(self.load()).decode("utf-8")

    def mime(self) -> str:
        """Get the MIME type for the image format.

        Returns:
            str: MIME type string (e.g., 'image/jpeg', 'image/png').
        """
        return f"image/{self.ext.lower()}" if self.ext else "image"

    def pil(self) -> ImagePil.Image:
        image = ImagePil.open(self.get_path())
        return image.rotate(self.orientation.to_degree(), expand=True)

    def _extract(self, path: Path) -> None:
        """Extract the image to a specified path."""
        if path.suffix == "":
            path = path.with_suffix(".jpg")
        self.pil().save(path)

    def check_valid_relative_crop_values(
        self,
        x_center: float,
        y_center: float,
        width_relative: float,
        height_relative: float,
    ):
        if not (0.0 <= x_center <= 1.0):
            raise ValueError(f"x_center is {x_center}")

        if not (0.0 <= y_center <= 1.0):
            raise ValueError(f"y_center is {y_center}")

        if not (0.0 <= width_relative <= 1.0):
            raise ValueError(f"width_relative is {width_relative}")

        if not (0.0 <= height_relative <= 1.0):
            raise ValueError(f"height_relative is {height_relative}")

    def make_crop(
        self,
        x_center: float,
        y_center: float,
        width_relative: float,
        height_relative: float,
        confidence: float | None = None,
        orientation: Orientation = Orientation.NORTH,
        name: str = "",
    ) -> "ImageCrop":

        self.check_valid_relative_crop_values(
            x_center,
            y_center,
            width_relative,
            height_relative,
        )

        return ImageCrop(
            name=name if name else self.name,
            data=self.data,
            x_center=x_center,
            y_center=y_center,
            width_relative=width_relative,
            height_relative=height_relative,
            confidence=confidence,
            orientation=orientation,
        )

    def make_crop_bounding_box(
        self,
        left,
        top,
        right,
        bottom,
        min_height: float = 0.0,
        confidence: float | None = None,
    ) -> "ImageCrop":
        bbox_width = right - left
        bbox_height = bottom - top
        x_center = (left + 0.5 * bbox_width) / self.width
        y_center = (top + 0.5 * bbox_height) / self.height
        width_relative = bbox_width / self.width
        height_relative = bbox_height / self.height

        return self.make_crop(
            x_center=x_center,
            y_center=y_center,
            width_relative=width_relative,
            height_relative=height_relative,
            confidence=confidence,
        )

    def evaluate(self, gold: "ImageCrop") -> float:
        raise NotImplementedError(
            "Base Image class does not implement evaluate(). Use ImageCrop instead."
        )

    def set_rel_to_src_parent(self) -> None:
        pass


@dataclass
class ImageCrop(Image):
    """Specialized image class for handling cropped regions of images.

    Extends the base Image class to support coordinate-based cropping with
    relative positioning. Coordinates are specified as normalized values
    (0.0 to 1.0) relative to the original image dimensions.

    Attributes:
        x_center (float): Normalized x-coordinate of crop center (0.0 to 1.0).
        y_center (float): Normalized y-coordinate of crop center (0.0 to 1.0).
        width_relative (float): Normalized width of crop region (0.0 to 1.0).
        height_relative (float): Normalized height of crop region (0.0 to 1.0).
    """

    x_center: float = field(default=0.5)
    y_center: float = field(default=0.5)
    width_relative: float = field(default=1.0)
    height_relative: float = field(default=1.0)
    confidence: float | None = field(default=None)
    source_parent: "Image | ImageCrop | None" = field(init=False, default=None)

    def save(self, path: Path | str = None):
        im = self.source_parent.pil() if self.source_parent else self.pil()
        if path is None:
            raise ValueError("Path must be provided to save the cropped image.")
        im.save(path)

    @property
    def width(self):
        return self.width_relative * self.raw_width

    @property
    def height(self):
        return self.height_relative * self.raw_height

    def add_source_parent(self, parent: "Image | ImageCrop") -> None:
        self.source_parent = parent

    def set_rel_to_src_parent(self) -> None:

        if not self.source_parent:
            return

        if type(self.source_parent) == ImageCrop:
            self.source_parent.set_rel_to_src_parent()
            dx = self.x_center - self.source_parent.x_center
            dy = self.y_center - self.source_parent.y_center
            self.x_center = 0.5 + dx / self.source_parent.width_relative
            self.y_center = 0.5 + dy / self.source_parent.height_relative
            self.width_relative /= self.source_parent.width_relative
            self.height_relative /= self.source_parent.height_relative

        match self.source_parent.orientation.to_degree():
            case -90:
                self.x_center, self.y_center = 1 - self.y_center, self.x_center
                self.width_relative, self.height_relative = (
                    self.height_relative,
                    self.width_relative,
                )
            case -180:
                self.x_center, self.y_center = 1 - self.x_center, 1 - self.y_center
            case -270:
                self.x_center, self.y_center = self.y_center, 1 - self.x_center
                self.width_relative, self.height_relative = (
                    self.height_relative,
                    self.width_relative,
                )

    def pil(self) -> ImagePil.Image:
        coordinates = self.coordinates()
        with ImagePil.open(self.get_path()) as imf:
            im_crop = imf.crop(coordinates)
        return im_crop.rotate(self.orientation.to_degree(), expand=True)

    def coordinates(self) -> tuple[float, float, float, float]:
        """Calculate absolute pixel coordinates for the crop region.

        Converts the normalized crop coordinates to absolute pixel coordinates
        within the bounds of the original image dimensions.

        Returns:
            tuple[float, float, float, float]: Absolute coordinates as
                (left, upper, right, bottom) in pixels.
        """
        x_center = self.x_center * self.raw_width
        y_center = self.y_center * self.raw_height
        actual_width_half = self.raw_width * self.width_relative / 2
        actual_height_half = self.raw_height * self.height_relative / 2
        left = max(0, min(int(x_center - actual_width_half), self.raw_width))
        upper = max(0, min(int(y_center - actual_height_half), self.raw_height))
        right = max(0, min(int(x_center + actual_width_half), self.raw_width))
        bottom = max(0, min(int(y_center + actual_height_half), self.raw_height))

        return (left, upper, right, bottom)

    def make_crop(
        self,
        x_center: float,
        y_center: float,
        width_relative: float,
        height_relative: float,
        confidence: float | None = None,
        orientation: Orientation = Orientation.NORTH,
        name: str = "",
    ) -> "ImageCrop":

        degree_transformed = orientation.to_degree()

        if degree_transformed == -90:
            x_center, y_center = y_center, 1 - x_center
            width_relative, height_relative = height_relative, width_relative
        elif degree_transformed == -180:
            x_center, y_center = 1 - x_center, 1 - y_center
        elif degree_transformed == -270:
            x_center, y_center = 1 - y_center, x_center
            width_relative, height_relative = height_relative, width_relative

        name = name if name else self.name

        return super().make_crop(
            x_center=self.x_center + (x_center - 0.5) * self.width_relative,
            y_center=self.y_center + (y_center - 0.5) * self.height_relative,
            width_relative=width_relative * self.width_relative,
            height_relative=height_relative * self.height_relative,
            confidence=confidence,
            orientation=orientation,
            name=name,
        )

    @staticmethod
    def compute_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if boxAArea + boxBArea - interArea == 0:
            return 0.0
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def evaluate(self, gold: "ImageCrop") -> float:
        if not isinstance(gold, ImageCrop):
            raise ValueError("Reference data must be an instance of ImageCrop.")
        iou = 0.0
        boxA = self.coordinates()
        boxB = gold.coordinates()
        iou = self.compute_iou(boxA, boxB)
        return iou
