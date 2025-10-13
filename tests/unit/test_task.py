import shutil

from collectra import Task, MachineLearningTask, ObjectDetectionYOLO
from collectra.utils import load_class_from_string
from utils.get_types import get_param_types, get_return_type

def test_generic_task(generic_type, debug):
    try:
        task = Task(name="generic_task")
        assert task.get_name() == "generic_task", "Task name should be set correctly"
        param_types = get_param_types(task.run)
        assert param_types == None, "Parameter types should be empty for base Task"
        return_type = get_return_type(task.run)        
        assert (
            return_type == generic_type
        ), "Return type should be Generic for base Task"
    except Exception as e:
        debug(e)


def test_generic_machine_learning_task(generic_type, debug):
    try:
        task = MachineLearningTask(name="ml_task")
        assert task.get_name() == "ml_task", "Task name should be set correctly"
        param_types = get_param_types(task.run)
        assert param_types == None, "Parameter types should be empty for base Task"
        return_type = get_return_type(task.run)
        assert (
            return_type == generic_type
        ), "Return type should be Generic for base Task"
    except Exception as e:
        debug(e)

def test_object_detection_task(task_data, images, train_yolo, classes, debug):
    try:
        name, data = task_data
        train_images, validation_images = images
        assert data[name], f"Task {name} should be in task_data"
        cls = load_class_from_string(data[name].pop("type"))
        task = cls(name=name, **data[name])        
        assert isinstance(task, ObjectDetectionYOLO), f"Task {name} should be an instance of ObjectDetectionYOLO"        
        results, log_dir = train_yolo(
            task=task,
            train_images=train_images,
            validation_images=validation_images,
            classes=classes)    
        assert results, "Training failed to return any results"
        assert results.results_dict is not None, "results_dict should exist"
        shutil.rmtree(log_dir, ignore_errors=True)  
    except Exception as e:
        debug(e)