import shutil
import traceback
from pathlib import Path

import pytest
from rich import print

from collectra import ImageCrop, ObjectDetectionYOLO
from collectra.utils import change_dir, error_msg


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
    from datetime import datetime

    from ultralytics.utils.metrics import DetMetrics

    def _train_yolo(
        task: ObjectDetectionYOLO, *images: ImageCrop, **kwargs
    ) -> DetMetrics | None:
        log = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # setting training run to be in the temp directory
        with change_dir(tmp_path):
            results = task.train(
                *images,
                log=log,
                **kwargs,
                base_folder=tmp_path,
            )
        return results

    return _train_yolo
