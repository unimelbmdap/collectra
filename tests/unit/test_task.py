from collectra.tasks.base import Task
from collectra.tasks.machine_learning.base import MachineLearningTask
from utils.get_types import get_param_types, get_return_type

def test_generic_task(generic_type):
    task = Task(name="generic_task")
    assert task.get_name() == "generic_task", "Task name should be set correctly"
    param_types = get_param_types(task.run)
    assert param_types == None, "Parameter types should be empty for base Task"
    return_type = get_return_type(task.run)
    assert return_type == generic_type, "Return type should be Generic for base Task"

def test_generic_machine_learning_task(generic_type):
    task = MachineLearningTask(name="ml_task")
    assert task.get_name() == "ml_task", "Task name should be set correctly"
    param_types = get_param_types(task.run)
    assert param_types == None, "Parameter types should be empty for base Task"
    return_type = get_return_type(task.run)
    assert return_type == generic_type, "Return type should be Generic for base Task"

# def test_object_detector(object_detector_input):
#     """
#         Test the object detector input.
#         - Ensure the input is an instance of Image.
#         - Ensure that task throws an error if the wrong type is provided.        

#     """

#     assert isinstance(object_detector_input, Image), "Input should be an instance of Image"
#     task = ObjectDetectionYOLO(name="label_detector")
#     param_types = get_param_types(task.run)
#     assert param_types, "Parameter types should not be None"
