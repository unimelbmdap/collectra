import sys
from types import ModuleType

import pytest

from collectra import ImageClassifierYOLO, ObjectDetectionYOLO


@pytest.mark.parametrize(
    ("task_class", "module_name"),
    [
        (
            ObjectDetectionYOLO,
            "collectra.tasks.object_detection.yolo.run_training_command",
        ),
        (
            ImageClassifierYOLO,
            "collectra.tasks.image_classifier.yolo.run_training_command",
        ),
    ],
)
@pytest.mark.parametrize("model", ["", "/models/base.pt"])
def test_train_forwards_optional_base_model(
    task_class, module_name, model, monkeypatch
):
    captured = {}

    def fake_training_command(task, inputs, trainer, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(module_name, fake_training_command)
    task = task_class("classifier", model="stored.pt")

    task.train([], model=model)

    assert captured["model"] == model
    assert task.model == "stored.pt"


@pytest.mark.parametrize("task_class", [ObjectDetectionYOLO, ImageClassifierYOLO])
@pytest.mark.parametrize(
    ("model_option", "expected"),
    [("", "stored.pt"), ("/models/base.pt", "/models/base.pt")],
)
def test_backend_training_uses_override_only_when_supplied(
    task_class, model_option, expected, monkeypatch
):
    fake_models = ModuleType("ultralytics.models")
    fake_models.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics.models", fake_models)
    task = task_class("classifier", model="stored.pt")
    observed = []

    def stop_after_model_selection():
        observed.append(task.model)
        raise RuntimeError("stop before backend initialization")

    monkeypatch.setattr(task, "_init_model", stop_after_model_selection)
    with pytest.raises(RuntimeError, match="stop before backend"):
        task._train(
            log="run",
            base_folder=None,
            model=model_option,
        )

    assert observed == [expected]
