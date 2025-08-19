import pytest
from collectra.task import Task, ObjectDetectionYOLO, ImageClassifier
from collectra.model import YOLOModel, DETECTRON2Engine, ImageClassifier

# def test_task_initialization():
#     try:
#         Task(task_type="test_task")
#         assert False, "Task should not be instantiated directly"
#     except Exception as e:
#         assert isinstance(e, TypeError), "Task should not be instantiated directly"

def test_detect_object_initialization():
    task = ObjectDetectionYOLO(task_type="detect_object")
    assert task.add_engine(YOLOModel()), "Engine should be added successfully"
    assert task.VALID_ENGINES == (YOLOModel, DETECTRON2Engine,), "Valid engines should include YOLOModel and DETECTRON2Engine"
    assert task.task_type == "detect_object", "Task type should be 'detect_object'"
    assert isinstance(task.engine, YOLOModel), "Engine should be an instance of YOLOModel"
    assert task.add_engine(ImageClassifier()) is False, "Invalid engine should not be added"
    engine = ImageClassifier()
    task = ObjectDetectionYOLO(task_type="detect_object", engine=engine)
    assert not isinstance(task.engine, ImageClassifier), "Engine should not be added to DetectObject task"
    task = ImageClassifier(task_type="classify_image", engine=engine)
    assert task.VALID_ENGINES == (ImageClassifier,), "Valid engines should only include ImageClassifier"
    assert isinstance(task.engine, ImageClassifier), "Engine should be an instance of ImageClassifier"

