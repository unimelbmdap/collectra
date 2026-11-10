"""Image segmentation tasks; model backends are imported only on execution."""

from .sam3 import ImageSegmenterSam3, ImageSegmentationSam3
from .rfdetr import ImageSegmenterRFDETR, ImageSegmentationRFDETR
from .yolo import ImageSegmenterYOLO, ImageSegmentationYOLO

__all__ = [
    "ImageSegmenterSam3",
    "ImageSegmenterRFDETR",
    "ImageSegmenterYOLO",
    # The former names.
    "ImageSegmentationSam3",
    "ImageSegmentationRFDETR",
    "ImageSegmentationYOLO",
]
