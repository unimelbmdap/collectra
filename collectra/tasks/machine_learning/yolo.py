"""Shared model lifecycle for YOLO-backed tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...logger import get_logger
from .base import MachineLearningTask

if TYPE_CHECKING:
    from ultralytics.models import YOLO

__all__ = ["YOLOTask"]

logger = get_logger(__name__)


class YOLOTask(MachineLearningTask):
    """Base class for tasks backed by an Ultralytics YOLO model."""

    model: str | Path | YOLO
    original_model_path: str | Path = ""

    def _init_model(self) -> None:
        """Load and validate the configured YOLO model."""
        from ultralytics.models import YOLO

        self._load()
        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance")

    def _reload(self) -> None:
        """Reload the model from its original path."""
        if not self.original_model_path:
            logger.warning("Original model path is not set. Skipping...")
            return
        self.model = self.original_model_path
        self._load()

    def _load(self) -> None:
        """Load a YOLO model only when execution or training requires it."""
        from ultralytics.models import YOLO

        if self.model and isinstance(self.model, (str, Path)):
            self.original_model_path = self.model
            self.model = YOLO(Path(self.model))
            return
        if not isinstance(self.model, YOLO):
            logger.warning("Invalid YOLO model value: %r", self.model)
