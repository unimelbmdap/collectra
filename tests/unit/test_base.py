"""Tests for collectra.commons.base module"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from collectra.commons.base import BaseEntity, Node, NodeStatus, TaskContext

# =============================================================================
# TaskContext
# =============================================================================


def test_taskcontext_defaults():
    ctx = TaskContext()
    assert ctx.usage_file is None
    assert ctx.render is False
    assert ctx.file_path is None
    assert ctx.single_run is False


def test_taskcontext_custom_values(tmp_path):
    f = tmp_path / "usage.yaml"
    ctx = TaskContext(usage_file=f, render=True, file_path=tmp_path, single_run=True)
    assert ctx.usage_file == f
    assert ctx.render is True
    assert ctx.file_path == tmp_path
    assert ctx.single_run is True


# =============================================================================
# NodeStatus
# =============================================================================


def test_nodestatus_values():
    assert NodeStatus.NOT_READY.value == "not_ready"
    assert NodeStatus.READY.value == "ready"
    assert NodeStatus.RUNNING.value == "running"
    assert NodeStatus.COMPLETED.value == "completed"
    assert NodeStatus.FAILED.value == "failed"


def test_nodestatus_str():
    assert str(NodeStatus.NOT_READY) == "not_ready"
    assert str(NodeStatus.COMPLETED) == "completed"
    assert str(NodeStatus.FAILED) == "failed"


# =============================================================================
# BaseEntity
# =============================================================================


@dataclass
class ConcreteEntity(BaseEntity):
    name: str

    def __call__(self):
        return self.name


def test_baseentity_name():
    e = ConcreteEntity(name="my_entity")
    assert e.get_name() == "my_entity"


def test_baseentity_str():
    e = ConcreteEntity(name="my_entity")
    result = str(e)
    assert "my_entity" in result
    assert "ConcreteEntity" in result


def test_baseentity_get_class_path():
    path = ConcreteEntity.get_class_path()
    assert path.endswith(".ConcreteEntity")


def test_baseentity_serialize_basic():
    e = ConcreteEntity(name="ent")
    serialized = e.serialize()
    assert "type" in serialized
    assert serialized["name"] == "ent"


def test_baseentity_serialize_path_becomes_name(tmp_path):
    e = ConcreteEntity(name="ent")
    e.some_path = tmp_path / "file.txt"
    serialized = e.serialize()
    assert serialized["some_path"] == "file.txt"


def test_baseentity_attributes_to_ignore_empty():
    e = ConcreteEntity(name="ent")
    assert e.attributes_to_ignore() == set()


# =============================================================================
# Node
# =============================================================================


def test_node_default_status():
    n = Node(name="n")
    assert n.status == NodeStatus.NOT_READY


def test_node_name():
    n = Node(name="my_node")
    assert n.name == "my_node"


def test_node_process_returns_empty_list():
    n = Node(name="n")
    assert n.process() == []
