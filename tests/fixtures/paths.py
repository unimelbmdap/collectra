from pathlib import Path

import pytest


@pytest.fixture
def root_data_path():
    return Path.cwd() / "tests" / "data"


@pytest.fixture
def raw_img_path(root_data_path):
    return root_data_path / "images" / "bar1.jpg"
