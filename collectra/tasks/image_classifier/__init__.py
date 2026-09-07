"""Image-classification task implementations."""

from .yolo import ImageClassifierYOLO
from .torchvision import ImageClassifierTorchvision

__all__ = ["ImageClassifierYOLO", "ImageClassifierTorchvision"]
