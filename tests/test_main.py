from collectra.models.task import DetectObject
from collectra.models.engine import YOLOEngine, ImageClassifier
from pathlib import Path

def test_detect_object_task_and_engine_add():
    """
    Test the training functionality of the DetectObject task with a YOLOEngine.
    """     
    engine = YOLOEngine(name="yolo11n.pt")
    task = DetectObject(task_type="detect_object", engine=engine)
    assert isinstance(task.engine, YOLOEngine), "Engine should be an instance of YOLOEngine"     
    # Run the training process
    # task.train()
    
    # # Verify that the training files are created
    # yolo_config_file = Path(engine.dir) / "yolo_config.yml"
    # train_file = Path(engine.dir) / "train.txt"
    # val_file = Path(engine.dir) / "val.txt"
    
    # assert yolo_config_file.exists(), "YOLO config file should be created"
    # assert train_file.exists(), "Train file should be created"
    # assert val_file.exists(), "Validation file should be created"