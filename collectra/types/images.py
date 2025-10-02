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


import base64, io, copy
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image as ImagePil

from collectra.utils import crop

from .base import Type


@dataclass(kw_only=False)
class Image(Type):
    """Base class for handling image data and metadata in Collectra workflows.
    
    Provides core functionality for loading, encoding, and managing image files
    with support for various formats and metadata extraction. Serves as the
    foundation for more specialized image processing classes.
    
    Attributes:
        path (str): File system path to the image file.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        format (str | None): Image format (PNG, JPEG, etc.) or None if unknown.
    """
    path: Path # Path to the image file
    width: int = field(init=False)  # Image width in pixels
    height: int = field(init=False) # Image height in pixels
    format: str | None = field(init=False, default=None) # Image format (e.g., PNG, JPEG)


    def __post_init__(self, **kwargs):
        """Create an Image instance from a file path.
        
        Loads image metadata (dimensions and format) from the specified file
        and returns a configured Image instance.
        
        Args:
            path (Path): Path to the image file to load.
            
        Returns:
            Image: Configured Image instance with loaded metadata.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            PIL.UnidentifiedImageError: If the file is not a valid image.
        """
        self.path = Path(self.path)
        if not self.path.exists() or not self.path.is_file():
            raise FileNotFoundError(f"Image file not found: {self.path}")
        
        with ImagePil.open(self.path) as imf:
            self.format = imf.format
            self.width, self.height = imf.size        
    
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
        
    def load(self) -> bytes:
        """Load the image file content as raw bytes.
        
        Returns:
            bytes: Raw binary content of the image file.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            PermissionError: If the file cannot be read due to permissions.
        """
        with open(self.path, "rb") as img_file:
            buffer = img_file.read()
        return buffer

    def get_encoding(self) -> str:
        """Get the base64 encoded representation of the image.
        
        Loads the image content and encodes it as a base64 string,
        suitable for embedding in JSON or other text-based formats.
        
        Returns:
            str: Base64 encoded string representation of the image.
        """
        buffer = self.load()
        return base64.b64encode(buffer).decode('utf-8')

    def mime(self) -> str:
        """Get the MIME type for the image format.
        
        Returns:
            str: MIME type string (e.g., 'image/jpeg', 'image/png').
        """
        return f"image/{self.format.lower()}" if self.format else "image"      

    def metadata(self) -> dict:
        """Extract metadata dictionary for the image.
        
        Returns:
            dict: Dictionary containing type information and file name.
        """
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}",  
            "path": Path(self.path).name,                      
        }    

@dataclass(kw_only=True)
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

    x_center: float
    y_center: float
    width_relative: float
    height_relative: float

    def coordinates(self) -> tuple[float, float, float, float]:
        """Calculate absolute pixel coordinates for the crop region.
        
        Converts the normalized crop coordinates to absolute pixel coordinates
        within the bounds of the original image dimensions.
        
        Returns:
            tuple[float, float, float, float]: Absolute coordinates as 
                (left, upper, right, bottom) in pixels.
        """
        x_center = self.x_center * self.width
        y_center = self.y_center * self.height
        actual_width_half = self.width * self.width_relative / 2
        actual_height_half = self.height * self.height_relative / 2
        left = max(0, min(int(x_center - actual_width_half), self.width))
        upper = max(0, min(int(y_center - actual_height_half), self.height))
        right = max(0, min(int(x_center + actual_width_half), self.width))
        bottom = max(0, min(int(y_center + actual_height_half), self.height))
        return (left, upper, right, bottom)

    def load(self) -> bytes:
        """Load the cropped region as raw bytes.
        
        Performs the actual crop operation using the calculated coordinates
        and returns the cropped image data as bytes.
        
        Returns:
            bytes: Raw binary content of the cropped image region.
        """
        img = crop(path=self.path, coordinates=self.coordinates())
        buffer = io.BytesIO()
        img.save(buffer, format=self.format)
        buffer.seek(0)        
        return buffer.read()
    
    def metadata(self) -> dict:
        """Extract metadata dictionary including crop coordinates.
        
        Returns:
            dict: Dictionary containing base metadata plus crop coordinates.
        """
        data = super().metadata()
        data.update({
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width_relative": self.width_relative,
            "height_relative": self.height_relative,
        })
        return data
    
    @staticmethod
    def metadata_list(images: list['ImageCrop']) -> dict:
        """Generate combined metadata for a list of ImageCrop instances.
        
        Creates a unified metadata structure for multiple crop regions from
        the same source image, useful for batch processing operations.
        
        Args:
            images (list[ImageCrop]): List of ImageCrop instances to process.
            
        Returns:
            dict: Combined metadata with list of crop coordinates.
            
        Raises:
            ValueError: If the images list is empty or contains invalid types.
        """
        if not images:
            raise ValueError("The images list is empty.")
        metadata = copy.deepcopy(images[0].metadata())
        list_to_pop = ["x_center", "y_center", "width_relative", "height_relative"]
        for key in list_to_pop:
            metadata.pop(key, None)
        metadata["items"] = list()
        for img in images:
            if not isinstance(img, ImageCrop):
                raise ValueError(f"Invalid image type. Expected ImageCrop, got {type(img)}")
            metadata["items"].append({
                "x_center": img.x_center,
                "y_center": img.y_center,
                "width_relative": img.width_relative,
                "height_relative": img.height_relative,
            })
        return metadata
