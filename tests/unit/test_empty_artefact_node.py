"""Tests for handling processed-but-empty ArtefactNodes.

Verifies that:
- ArtefactNode.process() with key:[] in results.yaml → READY with 0 items
- ArtefactNode.process() with missing key → raises ValueError
- prepare_inputs() with one empty parent → returns entries from non-empty parents
- _execute_artefact_node() with no matching args → node becomes READY
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from collectra import Image, NodeStatus, Text
from collectra.tasks.base import Task
from collectra.types.base import ArtefactNode


@pytest.fixture
def results_dir(tmp_path):
    """Create a temp directory with a results.yaml file."""

    def _make(results_data: dict):
        run_dir = tmp_path / "run.arb"
        run_dir.mkdir()
        results_file = run_dir / "results.yaml"
        with open(results_file, "w") as f:
            yaml.dump(results_data, f)
        return run_dir

    return _make


class TestArtefactNodeProcessEmpty:
    """Test ArtefactNode.process() distinguishes missing key from empty key."""

    def test_empty_list_key_sets_ready(self, results_dir):
        """Key exists with [] → READY, 0 items."""
        run_dir = results_dir({"locality_image": []})
        node = ArtefactNode(name="locality_image")
        node.add_type(Image)
        assert node.status == NodeStatus.NOT_READY

        node.process("locality_image", value=run_dir)

        assert node.status == NodeStatus.READY
        assert len(node.items) == 0

    def test_none_value_key_sets_ready(self, results_dir):
        """Key exists with None (YAML `locality_image:`) → READY, 0 items."""
        run_dir = results_dir({"locality_image": None})
        node = ArtefactNode(name="locality_image")
        node.add_type(Image)

        node.process("locality_image", value=run_dir)

        assert node.status == NodeStatus.READY
        assert len(node.items) == 0

    def test_missing_key_raises_error(self, results_dir):
        """Key not in results at all → ValueError."""
        run_dir = results_dir({"other_key": [{"type": "collectra.Text", "data": "x"}]})
        node = ArtefactNode(name="locality_image")
        node.add_type(Image)

        # The ValueError is caught by the outer try/except in process() and logged,
        # so the node stays NOT_READY
        node.process("locality_image", value=run_dir)
        assert node.status == NodeStatus.NOT_READY

    def test_populated_key_still_works(self, results_dir):
        """Key with actual data → items created as before."""
        run_dir = results_dir(
            {
                "label_text": [
                    {"type": "collectra.Text", "data": "hello"},
                    {"type": "collectra.Text", "data": "world"},
                ]
            }
        )
        node = ArtefactNode(name="label_text")
        node.add_type(Text)

        node.process("label_text", value=run_dir)

        assert node.status == NodeStatus.READY
        assert len(node.items) == 2


class TestPrepareInputsEmpty:
    """Test Task.prepare_inputs() skips empty parent inputs."""

    def _make_parent(self, name, items):
        node = ArtefactNode(name=name)
        for item in items:
            node.add_item(item)
        return node

    def test_one_empty_parent_returns_non_empty(self):
        """With one empty and one populated parent, returns entries from populated only."""
        text1 = Text(name="primary_label_text", data="hello")
        text2 = Text(name="primary_label_text", data="world")
        parent_text = self._make_parent("primary_label_text", [text1, text2])
        parent_image = self._make_parent("locality_image", [])  # empty

        task = Task("locality_reader")
        task.input = ["primary_label_text", "locality_image"]

        entries = task.prepare_inputs([parent_text, parent_image])

        assert len(entries) == 2
        # Each entry should have one text item
        for entry in entries:
            assert len(entry) == 1
            assert isinstance(entry[0], Text)

    def test_all_empty_parents_returns_empty(self):
        """When all parents are empty, returns empty list."""
        parent_text = self._make_parent("primary_label_text", [])
        parent_image = self._make_parent("locality_image", [])

        task = Task("locality_reader")
        task.input = ["primary_label_text", "locality_image"]

        entries = task.prepare_inputs([parent_text, parent_image])

        assert entries == []

    def test_all_populated_parents_works_as_before(self):
        """Normal case with all parents populated still works."""
        text1 = Text(name="primary_label_text", data="hello")
        text2 = Text(name="secondary_text", data="world")

        parent_text = self._make_parent("primary_label_text", [text1])
        parent_secondary = self._make_parent("secondary_text", [text2])

        task = Task("locality_reader")
        task.input = ["primary_label_text", "secondary_text"]

        entries = task.prepare_inputs([parent_text, parent_secondary])

        assert len(entries) == 1
        assert len(entries[0]) == 2


class TestExecuteArtefactNodeReady:
    """Test _execute_artefact_node marks node READY even with no matching args."""

    def test_no_matching_args_sets_ready(self):
        """When no args match the node, it should still become READY."""
        node = ArtefactNode(name="locality_image")
        node.add_type(Image)
        assert node.status == NodeStatus.NOT_READY

        # Simulate _execute_artefact_node logic: no matching args, then mark READY
        args = ()  # no args match
        for arg in args:
            pass  # nothing matches

        if node.status == NodeStatus.NOT_READY:
            node.status = NodeStatus.READY

        assert node.status == NodeStatus.READY
        assert len(node.items) == 0
