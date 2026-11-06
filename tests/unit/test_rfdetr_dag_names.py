"""The pipeline's output nodes are the source of truth for RF-DETR class names.

The classes sidecar is only a byproduct of training. The detector's outputs, in
the order the pipeline file lists them, name the model's classes. Whatever the
names, the model's last layer must have the matching number of outputs.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from collectra import Pipeline
from collectra.tasks.machine_learning.class_names import write_class_names
from collectra.tasks.object_detection import rfdetr as rfdetr_module
from collectra.tasks.object_detection.rfdetr import (
    ObjectDetectorRFDETR,
    class_head_size,
)

OUTPUTS = ["stamp", "label", "barcode"]


def fake_model(classes: int | None):
    """A stand-in for a loaded RF-DETR: its head has one extra (background) output."""
    head = None if classes is None else torch.nn.Linear(8, classes + 1)
    return SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(class_embed=head))
    )


@pytest.fixture
def checkpoint(tmp_path) -> Path:
    path = tmp_path / "weights" / "best.pt"
    path.parent.mkdir()
    path.write_bytes(b"not a real checkpoint")
    return path


def detector(checkpoint, monkeypatch, classes=len(OUTPUTS), outputs=OUTPUTS):
    """The detector task of a connected pipeline, loading a model with ``classes``."""
    monkeypatch.setattr(
        rfdetr_module, "load_rfdetr_model", lambda path: fake_model(classes)
    )
    pipeline = Pipeline(
        "Detect",
        "det",
        "0.1.0",
        path=str(checkpoint.parent.parent),
        detect={
            "type": "collectra.ObjectDetectorRFDETR",
            "model": str(checkpoint),
            "input": ["page"],
            "output": outputs,
        },
    )
    pipeline.connect()
    return pipeline.node_manager.resolve_node("detect").get_task()


def write_sidecar(checkpoint: Path, names: list[str]) -> None:
    write_class_names(checkpoint.parent / "classes.yaml", names)


def messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_class_head_size_is_the_number_of_outputs_of_the_last_layer():
    assert class_head_size(fake_model(3)) == 4


def test_class_head_size_is_unknown_for_a_model_without_the_layer():
    assert class_head_size(fake_model(None)) is None
    assert class_head_size(object()) is None


def test_names_come_from_the_pipeline_in_file_order_without_a_sidecar(
    checkpoint, monkeypatch, caplog
):
    task = detector(checkpoint, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=rfdetr_module.logger.name):
        task._load()
    assert task._categories == OUTPUTS  # not alphabetical
    assert not any("sidecar" in message for message in messages(caplog))


def test_a_sidecar_with_the_same_names_is_silent(checkpoint, monkeypatch, caplog):
    write_sidecar(checkpoint, OUTPUTS)
    task = detector(checkpoint, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=rfdetr_module.logger.name):
        task._load()
    assert task._categories == OUTPUTS
    assert messages(caplog) == []


def test_older_json_sidecar_is_still_compared(checkpoint, monkeypatch, caplog):
    (checkpoint.parent / "classes.json").write_text(json.dumps({"classes": ["a", "b"]}))
    task = detector(checkpoint, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=rfdetr_module.logger.name):
        task._load()
    assert task._categories == OUTPUTS
    assert "a, b" in messages(caplog)[0]


def test_pipeline_names_win_over_a_different_sidecar(checkpoint, monkeypatch, caplog):
    write_sidecar(checkpoint, ["a", "b", "c"])
    task = detector(checkpoint, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=rfdetr_module.logger.name):
        task._load()
    assert task._categories == OUTPUTS
    (warning,) = messages(caplog)
    assert "a, b, c" in warning and "stamp, label, barcode" in warning


def test_renaming_outputs_needs_no_new_model(checkpoint, monkeypatch):
    task = detector(checkpoint, monkeypatch, outputs=["x", "y", "z"])
    task._load()
    assert task._categories == ["x", "y", "z"]


@pytest.mark.parametrize("with_sidecar", [False, True])
def test_wrong_number_of_outputs_for_the_model_is_an_error(
    checkpoint, monkeypatch, with_sidecar
):
    if with_sidecar:
        write_sidecar(checkpoint, OUTPUTS)
    task = detector(checkpoint, monkeypatch, classes=5)
    with pytest.raises(ValueError, match="3 output.*5 classes"):
        task._load()


def test_unreadable_head_is_not_checked(checkpoint, monkeypatch):
    task = detector(checkpoint, monkeypatch, classes=None)
    task._load()
    assert task._categories == OUTPUTS


def test_reload_uses_the_pipeline_names_too(checkpoint, monkeypatch):
    write_sidecar(checkpoint, ["a", "b", "c"])
    task = detector(checkpoint, monkeypatch)
    task._load()
    task._categories = []
    task._reload()
    assert task._categories == OUTPUTS


class Variant:
    """A pretrained variant: its head has the 90 COCO classes."""

    def __new__(cls):
        return fake_model(90)


def test_pretrained_variant_keeps_its_own_class_names(checkpoint, monkeypatch):
    task = detector(checkpoint, monkeypatch)
    task.model = "nano"
    monkeypatch.setattr(
        rfdetr_module,
        "_backend_class",
        lambda name: type("Backend", (), {}) if name == "RFDETR" else Variant,
    )
    task._load()
    assert len(task._categories) == 80  # COCO, not the pipeline's three outputs
