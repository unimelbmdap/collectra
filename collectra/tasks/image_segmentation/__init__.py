"""Image segmentation tasks; model backends are imported only on execution."""

from .sam3 import ImageSegmentationSam3
from .rfdetr import ImageSegmentationRFDETR

__all__ = ["ImageSegmentationSam3", "ImageSegmentationRFDETR"]
