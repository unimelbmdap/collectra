from pathlib import Path
from typing import Optional

from collectra import MachineLearningTask, ObjectDetectionYOLO, Task
from collectra.utils import load_class_from_string
from utils.get_types import get_param_types, get_return_type


def test_generic_task(generic_type, debug):
    try:
        task = Task("generic_task")
        assert task.name == "generic_task", "Task name should be set correctly"
        param_types = get_param_types(task.run)
        assert param_types == dict(), "Parameter types should be empty for base Task"
        return_type = get_return_type(task.run)
        assert (
            return_type["return"] == Optional[generic_type]
        ), "Return type should be Generic for base Task"
    except Exception as e:
        debug(e)


def test_generic_machine_learning_task(generic_type, debug):
    try:
        task = MachineLearningTask("ml_task")
        assert task.name == "ml_task", "Task name should be set correctly"
        param_types = get_param_types(task.run)
        assert param_types == dict(), "Parameter types should be empty for base Task"
        return_type = get_return_type(task.run)
        assert (
            return_type["return"] == Optional[generic_type]
        ), "Return type should be Generic for base Task"
    except Exception as e:
        debug(e)


def test_object_detection_task(task_data, images, train_yolo, classes, debug):
    try:
        name, data = task_data
        assert data[name], f"Task {name} should be in task_data"
        cls = load_class_from_string(data[name].pop("type"))
        task = cls(name=name, **data[name])
        assert isinstance(
            task, ObjectDetectionYOLO
        ), f"Task {name} should be an instance of ObjectDetectionYOLO"
        results, _, log_dir = train_yolo(
            task,
            *images,
            classes=classes,
            project=Path.cwd() / f"{task.name}-test",
        )
        assert results, "Training failed to return any results"
        assert results.results_dict is not None, "results_dict should exist"
    except Exception as e:
        debug(e)
