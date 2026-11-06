"""Class names saved beside a checkpoint are YAML; the older JSON is still read."""

import json

import pytest
import yaml

from collectra.tasks.machine_learning.class_names import (
    read_class_names,
    sidecar_candidates,
    write_class_names,
)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"x")
    return path


def test_candidates_prefer_the_checkpoint_own_file_and_yaml(checkpoint):
    folder = checkpoint.parent
    assert sidecar_candidates(checkpoint) == [
        folder / "best.classes.yaml",
        folder / "best.classes.json",
        folder / "classes.yaml",
        folder / "classes.json",
    ]


def test_written_as_yaml_in_order(tmp_path):
    path = tmp_path / "classes.yaml"
    write_class_names(path, ["stamp", "label", "barcode"])
    assert yaml.safe_load(path.read_text()) == {
        "classes": ["stamp", "label", "barcode"]
    }
    assert path.read_text().index("stamp") < path.read_text().index("barcode")


def test_reads_yaml(checkpoint):
    write_class_names(checkpoint.parent / "classes.yaml", ["a", "b"])
    assert read_class_names(checkpoint) == ["a", "b"]


def test_still_reads_older_json(checkpoint):
    (checkpoint.parent / "classes.json").write_text(json.dumps({"classes": ["a", "b"]}))
    assert read_class_names(checkpoint) == ["a", "b"]


def test_checkpoint_own_file_beats_the_shared_one(checkpoint):
    write_class_names(checkpoint.parent / "classes.yaml", ["shared"])
    (checkpoint.parent / "best.classes.json").write_text(
        json.dumps({"classes": ["own"]})
    )
    assert read_class_names(checkpoint) == ["own"]


def test_yaml_beats_json_for_the_same_name(checkpoint):
    write_class_names(checkpoint.parent / "classes.yaml", ["new"])
    (checkpoint.parent / "classes.json").write_text(json.dumps({"classes": ["old"]}))
    assert read_class_names(checkpoint) == ["new"]


def test_none_when_there_is_no_sidecar_or_no_classes_in_it(checkpoint):
    assert read_class_names(checkpoint) is None
    (checkpoint.parent / "classes.yaml").write_text("other: 1\n")
    assert read_class_names(checkpoint) is None
