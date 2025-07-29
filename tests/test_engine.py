import pytest, os
from collectra.models.engine import Engine, YOLOEngine
from pathlib import Path

def test_yolo_engine_initialization():
    engine = YOLOEngine(name="test_yolo.pt")
    assert isinstance(engine, YOLOEngine), "Engine should be an instance of YOLOEngine"
    assert engine.name == Path("test_yolo.pt"), "Engine name should match the provided name"
    assert engine.config == {}, "Default config should be an empty dictionary"

def test_yolo_engine_run():
    config = {
        "data": "test_data",
        "file_format": "hespi"
    }
    engine = YOLOEngine(name="yolo11n.pt")
    engine.train(config=config)        
    yolo_config_file = Path(engine.dir) / "yolo_config.yml"
    train_file = Path(engine.dir) / "train.txt"
    val_file = Path(engine.dir) / "val.txt"
    assert yolo_config_file.exists(), "YOLO config file should be created"
    assert train_file.exists(), "Train file should be created"
    assert val_file.exists(), "Validation file should be created"
    
