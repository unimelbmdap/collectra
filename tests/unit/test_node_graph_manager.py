"""Tests for collectra.pipelines.node_graph_manager module"""

import pytest

from collectra import Text
from collectra.pipelines.node_graph_manager import NodeGraphManager
from collectra.tasks.base import Task, TaskNode
from collectra.types.base import DataNode


@pytest.fixture
def manager():
    return NodeGraphManager()


@pytest.fixture
def task():
    return Task("my_task")


# =============================================================================
# Graph lifecycle
# =============================================================================


def test_is_empty_on_init(manager):
    assert manager.is_empty()


def test_not_empty_after_adding_node(manager, task):
    manager.add_task_node("my_task", task)
    assert not manager.is_empty()


def test_copy_graph_returns_independent_copy(manager, task):
    manager.add_task_node("my_task", task)
    copy = manager.copy_graph()
    assert "my_task" in copy.nodes


# =============================================================================
# Node resolution
# =============================================================================


def test_resolve_node_not_found_raises(manager):
    with pytest.raises(ValueError):
        manager.resolve_node("missing")


def test_resolve_node_no_data_raises(manager):
    manager.flow.add_node("bare")
    with pytest.raises(ValueError):
        manager.resolve_node("bare")


# =============================================================================
# Node retrieval
# =============================================================================


def test_get_node_names(manager, task):
    manager.add_task_node("my_task", task)
    assert "my_task" in manager.get_node_names()


def test_get_task_nodes(manager, task):
    manager.add_task_node("my_task", task)
    nodes = manager.get_task_nodes()
    assert len(nodes) == 1
    assert isinstance(nodes[0], TaskNode)


def test_get_data_nodes(manager):
    manager.add_data_node("data_node", types=[Text])
    nodes = manager.get_data_nodes()
    assert len(nodes) == 1
    assert isinstance(nodes[0], DataNode)


# =============================================================================
# Parent / child retrieval
# =============================================================================


def test_get_parents_task(manager, task):
    manager.add_task_node("my_task", task)
    manager.add_data_node("output", types=[Text])
    manager.add_edge("my_task", "output")
    output_node = manager.resolve_node("output")
    parents = manager.get_parents_task(output_node)
    assert len(parents) == 1
    assert isinstance(parents[0], TaskNode)


def test_get_children_task(manager):
    task1 = Task("task1")
    task2 = Task("task2")
    manager.add_task_node("task1", task1)
    manager.add_task_node("task2", task2)
    manager.add_edge("task1", "task2")
    task1_node = manager.resolve_node("task1")
    children = manager.get_children_task(task1_node)
    assert len(children) == 1
    assert isinstance(children[0], TaskNode)


# =============================================================================
# add_data_node edge cases
# =============================================================================


def test_add_data_node_skips_when_stored_node_is_not_datanode(manager, task):
    """Node exists as TaskNode — add_data_node with obj should return early."""
    manager.add_task_node("shared", task)
    text = Text("shared", data="hello")
    manager.add_data_node("shared", obj=text)
    # No error and no item added to the task node
    task_node = manager.resolve_node("shared")
    assert isinstance(task_node, TaskNode)


def test_add_data_node_skips_when_type_not_in_node_types(manager):
    """obj type not in DataNode.types — item is not added."""
    from collectra import Image

    manager.add_data_node("data", types=[Image])
    text = Text("t", data="hello")
    before = len(manager.resolve_node("data").items)
    manager.add_data_node("data", obj=text, types=[Text])
    after = len(manager.resolve_node("data").items)
    assert before == after
