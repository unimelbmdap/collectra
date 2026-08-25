"""Public Collectra API, loaded on first attribute access."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BaseEntity": "collectra.commons.base",
    "Node": "collectra.commons.base",
    "NodeStatus": "collectra.commons.base",
    "TaskContext": "collectra.commons.base",
    "T": "collectra.commons.base",
    "Collectra": "collectra.pipelines.base",
    "Task": "collectra.tasks.base",
    "TaskNode": "collectra.tasks.base",
    "MachineLearningTask": "collectra.tasks.machine_learning.base",
    "ObjectDetectionYOLO": "collectra.tasks.machine_learning.yolo",
    "ImageClassifierYOLO": "collectra.tasks.machine_learning.yolo",
    "ObjectDetectionDETR": "collectra.tasks.machine_learning.detr",
    "DetectionTrainResult": "collectra.tasks.machine_learning.detr",
    "ObjectDetectionRFDETR": "collectra.tasks.machine_learning.rfdetr",
    "ImageOrienter": "collectra.tasks.machine_learning.orienters",
    "LLM": "collectra.tasks.llms",
    "LLMCanonicaliser": "collectra.tasks.canonicalisers",
    "IRNResolver": "collectra.tasks.irn_resolvers",
    "SuryaOCR": "collectra.tasks.ocr",
    "SuryaLineDetector": "collectra.tasks.ocr",
    "Data": "collectra.types.base",
    "DataNode": "collectra.types.base",
    "Image": "collectra.types.images",
    "ImageCrop": "collectra.types.images",
    "Orientation": "collectra.types.images",
    "Text": "collectra.types.texts",
    "Editor": "collectra.editor.base",
    "Evaluator": "collectra.evaluator.base",
    "generate_evaluation_html": "collectra.evaluator.visualise",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import public objects without eagerly importing optional ML backends."""
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
