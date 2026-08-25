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
    assert captured["epochs"] == 100
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


def test_classifier_train_forwards_regularization_options(monkeypatch):
    captured = {}

    def fake_training_command(task, inputs, trainer, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "collectra.tasks.image_classifier.yolo.run_training_command",
        fake_training_command,
    )
    task = ImageClassifierYOLO("classifier", model="stored-cls.pt")

    task.train(
        [],
        weight_decay=0.001,
        dropout=0.2,
        erasing=0.1,
        auto_augment="augmix",
        fliplr=0.25,
        flipud=0.1,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.3,
        cls_pw=0.5,
        freeze=8,
    )

    assert {
        name: captured[name]
        for name in (
            "weight_decay",
            "dropout",
            "erasing",
            "auto_augment",
            "fliplr",
            "flipud",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "cls_pw",
            "freeze",
        )
    } == {
        "weight_decay": 0.001,
        "dropout": 0.2,
        "erasing": 0.1,
        "auto_augment": "augmix",
        "fliplr": 0.25,
        "flipud": 0.1,
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.3,
        "cls_pw": 0.5,
        "freeze": 8,
    }


def test_classifier_prepare_params_forwards_regularization_options():
    task = ImageClassifierYOLO("classifier", model="stored-cls.pt")

    params = task._prepare_params(
        log="run",
        config_file="dataset",
        project="project",
        base_folder=None,
        weight_decay=0.001,
        dropout=0.2,
        erasing=0.1,
        auto_augment="augmix",
        fliplr=0.25,
        flipud=0.1,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.3,
        cls_pw=0.5,
        freeze=8,
    )

    assert params["weight_decay"] == 0.001
    assert params["dropout"] == 0.2
    assert params["erasing"] == 0.1
    assert params["auto_augment"] == "augmix"
    assert params["fliplr"] == 0.25
    assert params["flipud"] == 0.1
    assert params["hsv_h"] == 0.01
    assert params["hsv_s"] == 0.2
    assert params["hsv_v"] == 0.3
    assert params["cls_pw"] == 0.5
    assert params["freeze"] == 8

    params = task._prepare_params(
        log="run", config_file="dataset", project="project", base_folder=None
    )
    assert params["epochs"] == 100
    assert "freeze" not in params


def test_detector_prepare_params_defaults_to_100_epochs():
    task = ObjectDetectionYOLO("detector", model="stored.pt")

    params = task._prepare_params(
        log="run", config_file="dataset.yaml", project="project", base_folder=None
    )

    assert params["epochs"] == 100
