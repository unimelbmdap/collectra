"""Task classes with lazy optional-backend imports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Task": "collectra.tasks.base",
    "TaskNode": "collectra.tasks.base",
    "YOLOTask": "collectra.tasks.machine_learning.yolo",
    "ObjectDetectionYOLO": "collectra.tasks.object_detection.yolo",
    "ImageClassifierYOLO": "collectra.tasks.image_classifier.yolo",
    "ImageClassifierTorchvision": "collectra.tasks.image_classifier.torchvision",
    "ObjectDetectionDETR": "collectra.tasks.object_detection.detr",
    "DetectionTrainResult": "collectra.tasks.object_detection.detr",
    "ObjectDetectionRFDETR": "collectra.tasks.object_detection.rfdetr",
    "ImageOrienter": "collectra.tasks.machine_learning.orienters",
    "LLM": "collectra.tasks.llms",
    "LLMCanonicaliser": "collectra.tasks.canonicalisers",
    "IRNResolver": "collectra.tasks.irn_resolvers",
    "normalise_column_name": "collectra.tasks.irn_resolvers",
    "render_card": "collectra.tasks.irn_resolvers",
    "OCRSuraya": "collectra.tasks.ocr",
    "LineDetectorSurya": "collectra.tasks.ocr",
    "LineDetectorRLSA": "collectra.tasks.line_detection",
    "ConcatenateText": "collectra.tasks.text",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
