import pytest, yaml

from pathlib import Path

from collectra.types.images import Image, ImageCrop
from collectra.parsing import load_class_from_string


def convert_images(
    root_path: Path,
    key: str,
    value: dict,
    images: list[ImageCrop],
    validation_images: list[ImageCrop],
    validation: bool,
) -> None:
    if not ("type" in value and "path" in value):
        return
    cls = load_class_from_string(value.pop("type"))
    if cls == ImageCrop:
        value["name"] = key
        value["path"] = root_path / value["path"]
        img_crop = cls(**value)
        if img_crop:
            if validation:
                validation_images.append(img_crop)
            else:
                images.append(img_crop)


@pytest.fixture
def images(root_data_path) -> tuple[list[ImageCrop], list[ImageCrop]]:
    images_path = root_data_path / "images"
    images = list()
    validation_images = list()
    for image in images_path.glob("*.arb"):
        with open(image / "results.yaml", "r") as f:
            results: dict = yaml.safe_load(f)
            validation = results.pop("collectra_results_metadata")["validation"]
            for key, value in results.items():
                if isinstance(value, list):
                    for item in value:
                        convert_images(
                            image, key, item, images, validation_images, validation
                        )
                elif isinstance(value, dict):
                    convert_images(
                        image, key, value, images, validation_images, validation
                    )
    return images, validation_images


@pytest.fixture
def image(root_data_path) -> Image:
    return Image(name="specimen_sheet", path=root_data_path / "images" / "bar1.jpg")


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
            "output": classes
        }
    }       