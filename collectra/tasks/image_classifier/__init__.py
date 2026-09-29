"""Image-classification task implementations."""

from .yolo import ImageClassifierYOLO
from .torchvision import ImageClassifierTorchvision
from .huggingface import ImageClassifierHuggingFace

__all__ = [
    "ImageClassifierYOLO",
    "ImageClassifierTorchvision",
    "ImageClassifierHuggingFace",
]
