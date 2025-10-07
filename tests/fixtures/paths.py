import pytest
from pathlib import Path

@pytest.fixture
def root_data_path():
    return Path.cwd() / "tests" / "data"

@pytest.fixture
def model_path(root_data_path):
    return root_data_path / "dummy.pt"