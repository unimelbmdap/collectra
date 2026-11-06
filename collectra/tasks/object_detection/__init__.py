"""Object-detection task implementations."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ObjectDetector": "collectra.tasks.object_detection.base",
    "ObjectDetectionTask": "collectra.tasks.object_detection.base",  # the former name
    "ObjectDetectorYOLO": "collectra.tasks.object_detection.yolo",
    "ObjectDetectionYOLO": "collectra.tasks.object_detection.yolo",  # the former name
    "ObjectDetectorDETR": "collectra.tasks.object_detection.detr",
    "ObjectDetectionDETR": "collectra.tasks.object_detection.detr",  # the former name
    "DetectionTrainResult": "collectra.tasks.object_detection.detr",
    "ObjectDetectorRFDETR": "collectra.tasks.object_detection.rfdetr",
    "ObjectDetectionRFDETR": "collectra.tasks.object_detection.rfdetr",  # the former name
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
