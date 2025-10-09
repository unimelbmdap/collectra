from collectra.tasks.base import Task
from collectra.tasks.machine_learning.base import MachineLearningTask
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
