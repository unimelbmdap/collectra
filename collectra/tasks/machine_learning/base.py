"""Minimal base state for model-backed tasks."""

from __future__ import annotations

from typing import Generic

from collectra.commons import T

from ..base import Task

__all__ = ["MachineLearningTask"]


class MachineLearningTask(Task, Generic[T]):
    """A task that owns a model; capabilities are declared by subclasses."""

    model: T

    def __init__(self, name: str, model: T = None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.model = model
