"""Tests for DataNode.process() parsing from results.yaml.

Verifies that:
- Items with 'path' key are loaded the same as 'data' key
- 'parents' given as a single string is converted to a list
- partition from collectra_results_metadata is assigned to items
- Primitive (non-dict) values fall back to _create_instances
- A direct string value (not a directory) triggers the elif key: branch
"""

import pytest
import yaml

from collectra import NodeStatus, Text
from collectra.types.base import DataNode


@pytest.fixture
def results_dir(tmp_path):
    def _make(data: dict):
        run_dir = tmp_path / "run.arb"
        run_dir.mkdir()
        (run_dir / "results.yaml").write_text(yaml.dump(data))
        return run_dir

    return _make


class TestDataNodeProcessParsing:

    def test_path_key_loads_item(self, results_dir, tmp_path):
        """'path' key is treated the same as 'data' key."""
        txt = tmp_path / "sample.txt"
        txt.write_text("file content")
        run_dir = results_dir({"label": [{"type": "collectra.Text", "path": str(txt)}]})
        node = DataNode(name="label")
        node.add_type(Text)
        node.process("label", value=run_dir)
        assert node.status == NodeStatus.READY
        assert len(node.items) == 1

    def test_parents_single_string_becomes_list(self, results_dir):
        """parents given as a string is wrapped in a list."""
        run_dir = results_dir(
            {
                "label": [
                    {
                        "type": "collectra.Text",
                        "data": "hello",
                        "parents": "parent-id-abc",
                    }
                ]
            }
        )
        node = DataNode(name="label")
        node.add_type(Text)
        node.process("label", value=run_dir)
        item = list(node.items.values())[0]
        assert item.parents == ["parent-id-abc"]

    def test_partition_from_metadata_assigned_to_items(self, results_dir):
        """partition in collectra_results_metadata is set on each item."""
        run_dir = results_dir(
            {
                "collectra_results_metadata": {"partition": "train"},
                "label": [{"type": "collectra.Text", "data": "hello"}],
            }
        )
        node = DataNode(name="label")
        node.add_type(Text)
        node.process("label", value=run_dir)
        item = list(node.items.values())[0]
        assert item.partition == "train"

    def test_primitive_value_falls_back_to_create_instances(self, results_dir):
        """A plain string item (not a type dict) creates an instance via _create_instances."""
        run_dir = results_dir({"label": ["plain string"]})
        node = DataNode(name="label")
        node.add_type(Text)
        node.process("label", value=run_dir)
        assert node.status == NodeStatus.READY
        assert len(node.items) == 1

    def test_direct_string_value_creates_instance(self):
        """When value is not a directory, the elif key: branch creates instances directly."""
        node = DataNode(name="label")
        node.add_type(Text)
        node.process("label", value="some string value")
        assert node.status == NodeStatus.READY
        assert len(node.items) == 1
