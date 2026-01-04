import shutil
import traceback
from pathlib import Path

import pytest
from rich import print

from collectra import ImageCrop, ObjectDetectionYOLO
from collectra.utils import error_msg


@pytest.fixture
def debug(request):
    def debugger(e: Exception):
        print(error_msg(f"Test failed. Retaining all temporary dirs for debugging"))
        traceback.print_exc()
        if request.config.getoption("--pdb"):
            breakpoint()  # Debug here
        raise e

    return debugger


@pytest.fixture
def train_yolo(tmp_path):
    from ultralytics.utils.metrics import DetMetrics

    def _train_yolo(
        task: ObjectDetectionYOLO, *images: ImageCrop, **kwargs
    ) -> tuple[DetMetrics | None, DetMetrics | None, Path]:
        log_dir = tmp_path / "log_dir"
        Path(log_dir).mkdir(exist_ok=True)
        results, validation_results = task.train(
            *images,
            log_dir=log_dir,
            **kwargs,
        )
        return results, validation_results, log_dir

    return _train_yolo
