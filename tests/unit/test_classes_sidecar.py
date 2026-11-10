"""Tests for the class-names sidecar contract (YAML, with older JSON still read).

Two pieces of new behavior:

1. ``save_training_result`` copies ``classes.yaml`` from the training
   weights dir alongside the renamed checkpoint, named
   ``<new_model_stem>.classes.yaml``.

2. ``ObjectDetectorRFDETR._load_categories_from_sidecar`` looks up the
   sidecar at ``<checkpoint.stem>.classes.yaml`` first, then falls back
   to ``classes.yaml`` and the older ``.json`` names.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

# Local .venv has a corrupt opencv-python install; stub so the
# transitive ``from ultralytics.utils import ThreadingLocked`` import in
# rfdetr.py succeeds.
try:  # pragma: no cover - environment guard
    import cv2  # noqa: F401
except Exception:  # pragma: no cover - environment guard
    sys.modules["cv2"] = MagicMock()

from collectra.pipelines.base import Pipeline  # noqa: E402
from collectra.tasks.base import Task  # noqa: E402
from collectra.tasks.machine_learning.training import save_training_result  # noqa: E402
from collectra.tasks.object_detection.rfdetr import (  # noqa: E402
    ObjectDetectorRFDETR,
)

# ---------------------------------------------------------------------------
# Save-side: training utility copies classes.yaml next to renamed .pt
# ---------------------------------------------------------------------------


def _make_pipeline(task_name: str, current_model: str = "") -> Pipeline:
    return Pipeline("sidecar", "scr", "1", **{task_name: {"model": current_model}})


def test_save_train_copies_classes_sidecar(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"fake-pt")
    (weights / "classes.yaml").write_text("classes: [a, b]\n")

    results = SimpleNamespace(save_dir=tmp_path / "run", results_dict={})
    pipeline = _make_pipeline("rfdetr")
    task = Task(name="rfdetr", model="")
    task.pipeline = pipeline

    save_training_result(task, results, log="20260101_120000")

    new_pt = tmp_path / "rfdetr-20260101_120000.pt"
    sidecar = tmp_path / "rfdetr-20260101_120000.classes.yaml"
    assert new_pt.exists()
    assert sidecar.exists()
    assert yaml.safe_load(sidecar.read_text())["classes"] == ["a", "b"]


def test_save_train_converts_older_json_to_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"fake-pt")
    (weights / "classes.json").write_text(json.dumps({"classes": ["a", "b"]}))

    results = SimpleNamespace(save_dir=tmp_path / "run", results_dict={})
    task = Task(name="rfdetr", model="")
    task.pipeline = _make_pipeline("rfdetr")

    save_training_result(task, results, log="20260101_120000")

    sidecar = tmp_path / "rfdetr-20260101_120000.classes.yaml"
    assert yaml.safe_load(sidecar.read_text())["classes"] == ["a", "b"]
    assert list(tmp_path.glob("*.classes.json")) == []


def test_save_train_no_classes_json_no_crash(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"fake-pt")

    results = SimpleNamespace(save_dir=tmp_path / "run", results_dict={})
    pipeline = _make_pipeline("rfdetr")
    task = Task(name="rfdetr", model="")
    task.pipeline = pipeline

    save_training_result(task, results, log="20260101_120000")

    assert (tmp_path / "rfdetr-20260101_120000.pt").exists()
    assert list(tmp_path.glob("*.classes.*")) == []


# ---------------------------------------------------------------------------
# Load-side: ObjectDetectorRFDETR._load_categories_from_sidecar
# ---------------------------------------------------------------------------


@pytest.fixture
def task() -> ObjectDetectorRFDETR:
    return ObjectDetectorRFDETR(name="t")


def test_load_categories_prefers_stem_sidecar(
    tmp_path: Path, task: ObjectDetectorRFDETR
):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    (tmp_path / "best.classes.json").write_text(json.dumps({"classes": ["preferred"]}))
    (tmp_path / "classes.json").write_text(json.dumps({"classes": ["fallback"]}))

    task._load_categories_from_sidecar(ckpt)

    assert task._categories == ["preferred"]


def test_load_categories_falls_back_to_plain_classes_json(
    tmp_path: Path, task: ObjectDetectorRFDETR
):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    (tmp_path / "classes.json").write_text(json.dumps({"classes": ["a", "b"]}))

    task._load_categories_from_sidecar(ckpt)

    assert task._categories == ["a", "b"]


def test_load_categories_noop_when_missing(tmp_path: Path, task: ObjectDetectorRFDETR):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    task._categories = ["sentinel"]

    task._load_categories_from_sidecar(ckpt)

    assert task._categories == ["sentinel"]
