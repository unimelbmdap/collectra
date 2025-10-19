import pytest, yaml

from pathlib import Path

from collectra import Image, ImageCrop
from collectra.utils import change_dir, load_class_from_string

@pytest.fixture
def images(root_data_path) -> list[ImageCrop]:
    images_path = root_data_path / "images"
    images = list()    
    for image in images_path.glob("*.arb"):
        with change_dir(image):
            data = Path("results.yaml")
            if not data.exists():
                continue
            with open(data, "r") as f:
                results: dict = yaml.safe_load(f)
                validation = results.pop("collectra_results_metadata", dict()).get("validation", None)
            for key, value in results.items():
                result = list()
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if not isinstance(item, dict) or not ("type" in item and "path" in item):
                        continue                                        
                    cls_ = load_class_from_string(item.pop("type"))
                    if cls_ != ImageCrop:
                        continue
                    item["name"] = key
                    item["data"] = item.pop("path")                    
                    if validation is not None:
                        item["validation"] = validation
                    try:                         
                        instance = ImageCrop(**item)                                
                        if instance:
                            result.append(instance)
                    except Exception as e:
                        print(f"Failed to load data item {key} from {value}: {e}")
                        continue        
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
