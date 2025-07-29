import pytest
from collectra.models.task import Task, DetectObject, ClassifyImage
from collectra.models.engine import YOLOEngine, DETECTRON2Engine, ImageClassifier

def test_task_initialization():
    try:
        Task(task_type="test_task")
    except Exception as e:
        assert isinstance(e, TypeError), "Task should not be instantiated directly"

def test_detect_object_initialization():
    task = DetectObject(task_type="detect_object")
    assert task.add_engine(YOLOEngine()), "Engine should be added successfully"
    assert task.VALID_ENGINES == (YOLOEngine, DETECTRON2Engine,), "Valid engines should include YOLOEngine and DETECTRON2Engine"
    assert task.task_type == "detect_object", "Task type should be 'detect_object'"
    assert isinstance(task.engine, YOLOEngine), "Engine should be an instance of YOLOEngine"
    assert task.add_engine(ImageClassifier()) is False, "Invalid engine should not be added"
    engine = ImageClassifier()  
    task = DetectObject(task_type="detect_object", engine=engine)
    assert not isinstance(task.engine, ImageClassifier), "Engine should not be added to DetectObject task"
    task = ClassifyImage(task_type="classify_image", engine=engine)
    assert task.VALID_ENGINES == (ImageClassifier,), "Valid engines should only include ImageClassifier"
    assert isinstance(task.engine, ImageClassifier), "Engine should be an instance of ImageClassifier"

