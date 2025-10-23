import pytest, shutil, os
from pathlib import Path

pytest_plugins = [
    "tests.fixtures.functions",
    "tests.fixtures.paths",
    "tests.fixtures.models",
    "tests.fixtures.data",
    "tests.fixtures.types",
]


@pytest.fixture(autouse=True)
def cleanup_tmp_dir():
    shutil.rmtree("log_dir", ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_pt(root_data_path):
    file_pt = Path.cwd() / "yolo11n.pt"
    collectra_pt = root_data_path / "yolo11n.pt"
    if file_pt.exists():
        os.remove(file_pt)
    if collectra_pt.exists():
        os.remove(collectra_pt)
