import pytest
from collectra.models.task import Task, DetectObject
from collectra.models.engine import YOLOEngine, ImageClassifier

def test_task_initialization():
    try:
        Task(task_type="test_task")
    except Exception as e:
        assert isinstance(e, TypeError), "Task should not be instantiated directly"

def test_detect_object_initialization():
    task = DetectObject(task_type="detect_object")
    assert task.add_engine(YOLOEngine()), "Engine should be added successfully"
    assert task.task_type == "detect_object", "Task type should be 'detect_object'"
    assert isinstance(task.engine, YOLOEngine), "Engine should be an instance of YOLOEngine"
    assert task.add_engine(ImageClassifier()) is False, "Invalid engine should not be added"

