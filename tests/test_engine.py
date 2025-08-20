import pytest, os
from collectra.model import YOLOModel
from collectra.pipeline import CollectraManager
from pathlib import Path
import shutil


def test_yolo_model_initialization():
    model = YOLOModel(name="yolo11n.pt")
    assert isinstance(model, YOLOModel), "Model should be an instance of YOLOModel"
    assert model.name == Path("yolo11n.pt"), "Model name should match the provided name"


def test_yolo_model_run():
    config = {"input": "test_data", "file_format": "hespi"}
    model = YOLOModel(name="yolo11n.pt")
    os.makedirs(CollectraManager.TEMPORARY_DIR, exist_ok=True)
    model.preprocess(Path(config.get("input")), "hespi")
    yolo_config_file = Path(CollectraManager.TEMPORARY_DIR) / "yolo_config.yml"
    train_file = Path(CollectraManager.TEMPORARY_DIR) / "train.txt"
    val_file = Path(CollectraManager.TEMPORARY_DIR) / "val.txt"
    assert yolo_config_file.exists(), "YOLO config file should be created"
    assert train_file.exists(), "Train file should be created"
    assert val_file.exists(), "Validation file should be created"
    shutil.rmtree(CollectraManager.TEMPORARY_DIR, ignore_errors=True)
