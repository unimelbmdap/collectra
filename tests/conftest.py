from pathlib import Path    
import pytest

image_path = Path.cwd() / "tests/data/label.hespi"

@pytest.fixture
def image_data_path():
    return image_path

@pytest.fixture
def workflow():
    return Path.cwd() / "samples/Hespiv2"

@pytest.fixture
def img():
    return image_path, image_path / "label.png"

@pytest.fixture
def cropped():
    field = "locality"
    return field, image_path / field / "im.jpg"