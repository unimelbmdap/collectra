"""Machine-learning task classes with lazy backend imports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "YOLOTask": "collectra.tasks.machine_learning.yolo",
    "ImageOrienter": "collectra.tasks.machine_learning.orienters",
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
