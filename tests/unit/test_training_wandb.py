"""Experiment tracking is explicitly enabled per training command."""

import importlib
from types import ModuleType, SimpleNamespace

import pytest

from collectra.cli import invoke


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("image_classifier.torchvision", "ImageClassifierTorchvision"),
        ("image_classifier.huggingface", "ImageClassifierHuggingFace"),
        ("image_classifier.yolo", "ImageClassifierYOLO"),
        ("object_detection.yolo", "ObjectDetectionYOLO"),
        ("object_detection.detr", "ObjectDetectionDETR"),
        ("object_detection.rfdetr", "ObjectDetectionRFDETR"),
    ],
)
def test_training_cli_wandb_is_opt_in(module_name, class_name, monkeypatch):
    module = importlib.import_module(f"collectra.tasks.{module_name}")
    captured = []
    monkeypatch.setattr(
        module,
        "run_training_command",
        lambda *args, **kwargs: captured.append(kwargs["wandb"]),
    )
    task = getattr(module, class_name)("task")
    # Enabling one run must not enable the next one by default.
    for options in ([], ["--wandb"], [], ["--no-wandb"]):
        invoke(task, ["train", "data", *options])
    assert captured == [False, True, False, False]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_yolo_wandb_scoped_to_training(enabled, fails, monkeypatch):
    import ultralytics
    from collectra.tasks.machine_learning.yolo import YOLOTask

    settings = {"wandb": not enabled}
    monkeypatch.setattr(ultralytics, "settings", settings)
    integration = ModuleType("ultralytics.utils.callbacks.wb")

    def wandb_callback():
        pass

    wandb_callback.__module__ = integration.__name__
    other_callback = lambda: None

    def reload_integration(module):
        assert module is integration
        module.callbacks = {"event": wandb_callback} if settings["wandb"] else {}
        return module

    monkeypatch.setattr(importlib, "import_module", lambda name: integration)
    monkeypatch.setattr(importlib, "reload", reload_integration)
    task = YOLOTask("task")
    task.model = SimpleNamespace(callbacks={"event": [other_callback, wandb_callback]})
    try:
        with task._wandb_logging(enabled):
            assert settings["wandb"] is enabled
            assert bool(integration.callbacks) is enabled
            assert task.model.callbacks["event"] == [other_callback]
            task.model.callbacks["event"].extend(integration.callbacks.values())
            if fails:
                raise RuntimeError("training failed")
    except RuntimeError as error:
        assert fails and str(error) == "training failed"
    assert settings["wandb"] is not enabled
    assert bool(integration.callbacks) is not enabled
    assert task.model.callbacks["event"] == [other_callback]


def test_rfdetr_wandb_backend_defaults_off(tmp_path):
    from collectra import ObjectDetectionRFDETR

    task = ObjectDetectionRFDETR("detector")
    assert task._prepare_params(output_dir=tmp_path)["wandb"] is False
    assert task._prepare_params(output_dir=tmp_path, wandb=True)["wandb"] is True
