import pytest, yaml

from pathlib import Path

from collectra.types.images import Image, ImageCrop
from collectra.utils import change_dir

@pytest.fixture
def images(root_data_path) -> list[ImageCrop]:
    images_path = root_data_path / "images"
    images = list()    
    for image in images_path.glob("*.arb"):
        with change_dir(image):
            result = Image.handle(image)
        images.extend(result)
    return images


@pytest.fixture
def image(raw_img_path) -> Image:
    return Image(name="specimen_sheet", data=raw_img_path)


@pytest.fixture
def classes() -> list[str]:
    return ["human", "chair", "table", "car"]


@pytest.fixture
def model(root_data_path) -> Path:
    return root_data_path / "yolo11n.pt"


@pytest.fixture
def pipeline(root_data_path) -> dict:
    pipeline_path = root_data_path / "collectrapipeline" / "pipeline.yaml"
    with open(pipeline_path, "r") as f:
        config: dict = yaml.safe_load(f)
    config["pipeline_path"] = pipeline_path.parent
    return config


@pytest.fixture
def task_data(classes) -> tuple[str, dict]:
    task_name = "label_detector"
    return task_name, {
        f"{task_name}": {
            "type": "collectra.ObjectDetectionYOLO",
            "model": "yolo11n.pt",
            "input": "specimen_sheet",
            "epochs": 1,
            "output": classes,
        }
    }
