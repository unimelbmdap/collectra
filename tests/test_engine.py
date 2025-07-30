import pytest, os
from collectra.models.engine import YOLOEngine
from collectra.models.workflow import Collectra
from pathlib import Path
import shutil

def test_yolo_engine_initialization():
    engine = YOLOEngine(name="yolo11n.pt")
    assert isinstance(engine, YOLOEngine), "Engine should be an instance of YOLOEngine"
    assert engine.name == Path("yolo11n.pt"), "Engine name should match the provided name"    

def test_yolo_engine_run():
    config = {
        "input": "test_data",        
        "file_format": "hespi"
    }
    engine = YOLOEngine(name="yolo11n.pt")
    os.makedirs(Collectra.TEMPORARY_DIR, exist_ok=True)
    engine.preprocess(Path(config.get("input")), "hespi")
    yolo_config_file = Path(Collectra.TEMPORARY_DIR) / "yolo_config.yml"
    train_file = Path(Collectra.TEMPORARY_DIR) / "train.txt"
    val_file = Path(Collectra.TEMPORARY_DIR) / "val.txt"
    assert yolo_config_file.exists(), "YOLO config file should be created"
    assert train_file.exists(), "Train file should be created"
    assert val_file.exists(), "Validation file should be created"
    shutil.rmtree(Collectra.TEMPORARY_DIR, ignore_errors=True)
    
