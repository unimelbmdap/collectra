"""Tests for collectra.types.base Data and DataNode methods"""

import pytest

from collectra import Image, Text
from collectra.types.base import Data, DataNode


class ConcreteData(Data):
    def __call__(self):
        return self.name


class OtherData(Data):
    def __call__(self):
        return self.name

    def evaluate(self, gold) -> float:
        return 1.0


# =============================================================================
# Data.serialize
# =============================================================================


def test_data_serialize_removes_empty_parents():
    item = Text("label", data="hello")
    serialized = item.serialize()
    assert "parents" not in serialized


def test_data_serialize_collapses_single_parent():
    item = Text("label", data="hello")
    parent = Text("parent", data="p")
    item.parents = [parent.id]
    serialized = item.serialize()
    assert serialized["parents"] == parent.id


# =============================================================================
# Data.evaluate
# =============================================================================


def test_data_evaluate_raises_not_implemented():
    item = ConcreteData(name="test")
    with pytest.raises(NotImplementedError):
        item.evaluate(None)


# =============================================================================
# DataNode.get_item
# =============================================================================


def test_get_item_returns_existing():
    node = DataNode(name="labels")
    item = Text("labels", data="hello")
    node.add_item(item)
    assert node.get_item(item.id) is item


def test_get_item_returns_none_for_missing():
    node = DataNode(name="labels")
    assert node.get_item("nonexistent-id") is None


# =============================================================================
# DataNode.check_type
# =============================================================================


def test_check_type_returns_false_when_not_registered():
    node = DataNode(name="labels")
    node.add_type(Text)
    assert node.check_type(Image) is False


# =============================================================================
# DataNode.evaluate
# =============================================================================


def test_evaluate_raises_type_error_on_incompatible_types():
    predicted = DataNode(name="pred")
    predicted.add_item(Text("label", data="hello"))

    gold = DataNode(name="gold")
    gold.add_item(OtherData(name="label"))

    with pytest.raises(TypeError):
        predicted.evaluate(gold)


def test_evaluate_more_gold_than_predicted():
    predicted = DataNode(name="pred")
    predicted.add_item(Text("label", data="hello"))

    gold = DataNode(name="gold")
    gold.add_item(Text("label", data="hello"))
    gold.add_item(Text("label", data="world"))
    gold.add_item(Text("label", data="foo"))

    result = predicted.evaluate(gold)
    assert result["num_predicted"] == 1
    assert result["num_gold"] == 3


def test_evaluate_more_predicted_than_gold():
    predicted = DataNode(name="pred")
    predicted.add_item(Text("label", data="hello"))
    predicted.add_item(Text("label", data="world"))
    predicted.add_item(Text("label", data="foo"))

    gold = DataNode(name="gold")
    gold.add_item(Text("label", data="hello"))

    result = predicted.evaluate(gold)
    assert result["num_predicted"] == 3
    assert result["num_gold"] == 1
