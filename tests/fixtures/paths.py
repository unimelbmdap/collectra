import pytest
from pathlib import Path


@pytest.fixture
def root_data_path():
    return Path.cwd() / "tests" / "data"
