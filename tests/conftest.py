import os
from pathlib import Path

import pytest

# Disable experiment tracking before test collection imports any ML backends.
# Subprocesses inherit these settings, including CLI and DataLoader tests.
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

pytest_plugins = [
    "tests.fixtures.functions",
    "tests.fixtures.paths",
    "tests.fixtures.models",
    "tests.fixtures.data",
    "tests.fixtures.types",
    "tests.fixtures.ensemble",
    "tests.fixtures.converters",
]


@pytest.fixture(autouse=True)
def cleanup_pt(root_data_path):
    file_pt = Path.cwd() / "yolo11n.pt"
    collectra_pt = root_data_path / "yolo11n.pt"
    if file_pt.exists():
        os.remove(file_pt)
    if collectra_pt.exists():
        os.remove(collectra_pt)
