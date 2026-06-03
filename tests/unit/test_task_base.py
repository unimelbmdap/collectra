"""Tests for collectra.tasks.base module"""

import pytest

from collectra import Text
from collectra.tasks.base import Task, TaskNode
from collectra.types.base import DataNode

# =============================================================================
# Task.input_dict
# =============================================================================


def test_input_dict_single_string_input():
    task = Task("t", input="label")
    assert task.input_dict == {"label": []}


def test_input_dict_list_input():
    task = Task("t", input=["label", "image"])
    assert task.input_dict == {"label": [], "image": []}


def test_input_dict_no_input_attr():
    task = Task("t")
    assert task.input_dict == {}


# =============================================================================
# Task.run / Task.__call__
# =============================================================================


def test_run_raises_not_implemented():
    task = Task("t")
    with pytest.raises(NotImplementedError):
        task.run()


def test_call_delegates_to_run():
    task = Task("t")
    with pytest.raises(NotImplementedError):
        task()


# =============================================================================
# Task.get_output_name
# =============================================================================


def test_get_output_name_default():
    task = Task("my_task")
    assert task.get_output_name() == "my_task_output"


def test_get_output_name_list_output():
    task = Task("my_task", output=["result"])
    assert task.get_output_name() == "result"


def test_get_output_name_string_output():
    task = Task("my_task", output="result")
    assert task.get_output_name() == "result"


# =============================================================================
# Task.prepare_inputs — parent validation
# =============================================================================


def test_prepare_inputs_filters_mismatched_parents():
    """Cross-paired entries are removed when item.parents references a different parent."""
    text1 = Text("images", data="img1")
    text2 = Text("images", data="img2")

    crop1 = Text("crops", data="crop1")
    crop1.parents = [text1.id]

    crop2 = Text("crops", data="crop2")
    crop2.parents = [text2.id]

    parent_images = DataNode(name="images")
    parent_images.add_item(text1)
    parent_images.add_item(text2)

    parent_crops = DataNode(name="crops")
    parent_crops.add_item(crop1)
    parent_crops.add_item(crop2)

    task = Task("reader", input=["images", "crops"])
    entries = task.prepare_inputs([parent_images, parent_crops])

    assert len(entries) == 2
    for entry in entries:
        image_item = next(e for e in entry if e.name == "images")
        crop_item = next(e for e in entry if e.name == "crops")
        assert image_item.id in crop_item.parents


# =============================================================================
# TaskNode
# =============================================================================


def test_tasknode_get_task():
    task = Task("t")
    node = TaskNode(name="t", task=task)
    assert node.get_task() is task
